package io.bebebus.livebeauty

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * 基于 MediaPipe 478 真实关键点的 2D 三角网格精细微整 CPU 变形器（仅 Android，纯逻辑无 GL 依赖）。
 *
 * 特点：
 * - 全部 14 项秀场微整（含 6 项双向调节）在度量空间 `m = (x, y/aspect)` 内以真实解剖关键点为锚点计算平滑位移，
 *   并通过 `mirrorSign` 对正常与水平镜像点集保持严格方向不变；
 * - 3D 透视/转头自遮挡可见度是连续量：三角形投影面积 ÷（投影尺度² × 规范 3D 面积）就是法线与视线夹角的余弦 ρ，
 *   取 smoothstep(0, [RHO_FADE], ρ)，按面积加权到顶点后平方、再做 2 轮一环拓扑平滑；投影尺度取弱透视姿态的 |r1 × r2|
 *   （见 [FaceMeshFilter.viewScale2]），与偏航无关（脸宽 × 脸高或 |det A| 会随转头按 cos 缩小，把远侧可见度高估）；
 *   以前按有向面积 1e-6 二值判定，下颌与眼睑处的细长三角形随噪声来回翻转，形变逐帧跳 5–10 px；
 * - 对多滑杆叠加内收位移施加 C∞ 有理软饱和上限（`maxInward = 0.120 * fw`）与一阶拓扑拉普拉斯平滑，杜绝凹陷咬脸与锯齿裂缝；
 * - 外轮廓缓冲环（顶点 468..503）与屏幕外锚点（504..511）位移严格恒为 0，确保背后墙面、门框直线零弯曲；
 * - 防翻转守卫是连续的障碍函数：固定 [RELAX_ITER] 轮，按面积比平滑收缩（不提前退出、无阈值跳变），只约束原本正面的三角形，
 *   邻域平滑松弛系数，彻底不做单点硬归零（v4.1「脸缺半块」根因 1）；
 * - 传入 `nowMs` 时可见度与松弛系数做时间平滑（可见度下降 40 ms、上升 120 ms；松弛变小立即生效、变大 120 ms），
 *   松弛的时间平滑作为防翻转守卫的起点（上限），最终绘制的系数仍经守卫校验；不传时为无状态（单测与离线复算）；
 * - 全程复用预分配数组，每帧 0 堆分配。
 *
 * 三个通道（FaceMeshWarpPass、SkinMaskPass、MakeupPass）的硬件背面剔除保持不变，可见度渐隐不能替代它：没有深度缓冲时，
 * 折叠区是正面加背面的二重覆盖，背面三角形即使位移渐隐到 0，仍按自己的源坐标采样原图、按索引顺序盖到已变形的正面上，
 * 形成接缝，美妆还会把修容印到鼻梁上（v4.1 根因 2）。剔除本身不是抖动源：三角形只在有向面积过零时切换剔除状态，
 * 此时它覆盖的面积趋于 0，切换的覆盖面积随噪声连续；抖动主要来自旧的二值可见度与 relax ×0.65 的位移跳变。
 */
internal class FaceMeshDeformer {
    /** 15 项微整输入参数（14 项五官脸型 + 末尾新增高颅顶蓬松头包脸 cranialTop，保持位置参数兼容）。 */
    data class Params(
        val faceSlim: Float = 0f,
        val vFace: Float = 0f,
        val faceNarrow: Float = 0f,
        val faceSmall: Float = 0f,
        val cheekboneSlim: Float = 0f,
        val chinShape: Float = 0f,       // -1..+1（正拉长尖细，负缩短）
        val foreheadShape: Float = 0f,   // -1..+1（正抬高发际线，负降低发际线）
        val eyeEnlarge: Float = 0f,      // 0..1
        val eyeDistance: Float = 0f,     // -1..+1（正拉宽眼距，负拉近眼距）
        val eyeCorner: Float = 0f,       // -1..+1（正上扬猫眼，负下垂无辜眼）
        val noseSlim: Float = 0f,        // 0..1
        val noseLength: Float = 0f,      // -1..+1（正拉长，负缩短）
        val mouthSize: Float = 0f,       // -1..+1（正放大，负缩小）
        val mouthSmile: Float = 0f,      // 0..1
        val cranialTop: Float = 0f,      // 0..1（高颅顶/蓬松头包脸：保持发际线不拉长裸额头，外蓬颅顶与上颞发冠）
    ) {
        fun isActive(): Boolean =
            faceSlim > EPS || vFace > EPS || faceNarrow > EPS || faceSmall > EPS || cheekboneSlim > EPS ||
                abs(chinShape) > EPS || abs(foreheadShape) > EPS || cranialTop > EPS || eyeEnlarge > EPS ||
                abs(eyeDistance) > EPS || abs(eyeCorner) > EPS || noseSlim > EPS ||
                abs(noseLength) > EPS || abs(mouthSize) > EPS || mouthSmile > EPS

        companion object {
            private const val EPS = 1e-4f
        }
    }

    /** GPU 顶点缓冲：512 × 4 = 2048 floats，排布为 `(dstNdcX, dstNdcY, srcTexX, srcTexY)`。 */
    val vertexData = FloatArray(FaceMeshTopology.TOTAL_VERTEX_COUNT * 4)

    /** 变形后的完整 478 点纹理坐标（y 向上），供局部修饰、多边形皮肤遮罩与实时美妆复用。 */
    val warpedLandmarks = FloatArray(FaceMeshFrame.COORD_COUNT)

    /** 最近一次 [deform] 的 468 点可见度（0..1，含时间平滑），只读，单测用。 */
    internal val vertexVisibility: FloatArray get() = vis

    /** 最近一次 [deform] 的正面绕序是否为顺时针（常规点集 false = CCW，水平镜像点集 true = CW）。 */
    var frontFaceCw: Boolean = false
        private set

    // 预分配工作缓冲（度量空间），零运行期 GC。
    private val srcMx = FloatArray(FaceMeshTopology.TOTAL_VERTEX_COUNT)
    private val srcMy = FloatArray(FaceMeshTopology.TOTAL_VERTEX_COUNT)
    private val dstMx = FloatArray(FaceMeshTopology.TOTAL_VERTEX_COUNT)
    private val dstMy = FloatArray(FaceMeshTopology.TOTAL_VERTEX_COUNT)
    private val dispMx = FloatArray(FaceMeshTopology.BUFFER_OFFSET)
    private val dispMy = FloatArray(FaceMeshTopology.BUFFER_OFFSET)
    private val interiorMx = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val interiorMy = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val smoothMx = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val smoothMy = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val vis = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val visScratch = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val relax = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val relaxScratch = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val relaxFactor = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val pose = FloatArray(FaceMeshFilter.POSE_SIZE)
    private val triVis = FloatArray(FaceMeshTopology.CANONICAL_TRIANGLE_COUNT)
    private val triFront = FloatArray(FaceMeshTopology.DEFORMABLE_TRIANGLE_COUNT)
    private val triOrig = FloatArray(FaceMeshTopology.DEFORMABLE_TRIANGLE_COUNT)
    // 时间平滑状态（只在传入 nowMs 时使用）。
    private val visState = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private val relaxState = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
    private var temporalValid = false
    private var lastNowMs = Long.MIN_VALUE

    /** 丢弃可见度与松弛系数的时间平滑状态（新一轮、换人、重新检出时调用）。 */
    fun resetTemporal() {
        temporalValid = false
        lastNowMs = Long.MIN_VALUE
    }

    /**
     * 对 [landmarks]（478×2 GL 纹理坐标）执行 14 项网格微整，结果写入 [vertexData] 与 [warpedLandmarks]。
     * [nowMs] 为本帧渲染时间：传入时可见度与松弛系数做时间平滑（距上次超过 [TEMPORAL_RESET_MS] 自动重置）；
     * 缺省 [Long.MIN_VALUE] 为无状态模式。
     */
    fun deform(
        landmarks: FloatArray,
        aspect: Float,
        axisX: Float,
        axisY: Float,
        faceWidth: Float,
        faceHeight: Float,
        leftWeight: Float,
        rightWeight: Float,
        yawWeight: Float,
        presence: Float,
        steadyFace: Float,
        steadyEye: Float,
        params: Params,
        nowMs: Long = Long.MIN_VALUE,
    ): Boolean {
        if (landmarks.size < FaceMeshFrame.COORD_COUNT || !(aspect > 0f) || faceWidth < 0.03f || faceHeight < 0.03f) {
            return false
        }
        // 时间平滑：距上次调用不超过 TEMPORAL_RESET_MS 才沿用状态，否则本帧重新初始化。
        val temporal = nowMs != Long.MIN_VALUE
        var dtMs = 0f
        var temporalFresh = true
        if (temporal) {
            if (temporalValid && lastNowMs != Long.MIN_VALUE && nowMs >= lastNowMs && nowMs - lastNowMs <= TEMPORAL_RESET_MS) {
                dtMs = (nowMs - lastNowMs).toFloat()
                temporalFresh = false
            }
            lastNowMs = nowMs
        }
        val axisNorm = hypot(axisX, axisY)
        if (axisNorm < 1e-4f) return false
        val ux = axisX / axisNorm; val uy = axisY / axisNorm
        val vx = -uy; val vy = ux

        // 1. 转换前 468 个关键点到度量空间 m = (x, y / aspect)。
        for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
            srcMx[i] = landmarks[i * 2]
            srcMy[i] = landmarks[i * 2 + 1] / aspect
            dispMx[i] = 0f
            dispMy[i] = 0f
            interiorMx[i] = 0f
            interiorMy[i] = 0f
        }
        for (i in FaceMeshTopology.FACE_VERTEX_COUNT until FaceMeshTopology.BUFFER_OFFSET) {
            dispMx[i] = 0f
            dispMy[i] = 0f
        }

        // 检测关键点横向极性：常规图像 (m454 - m234)·u >= 0 为 +1（CCW 正面），水平翻转点集为 -1（CW 正面）。
        val mirrorSign = if ((srcMx[454] - srcMx[234]) * ux + (srcMy[454] - srcMy[234]) * uy >= 0f) 1f else -1f
        frontFaceCw = mirrorSign < 0f

        // 3D 透视/转头自遮挡可见度 vis[0..467]：连续的 ρ = 投影面积 / (s² · 规范 3D 面积)，背面自遮挡区渐隐到 0。
        val indices = FaceMeshTopology.FULL_INDICES
        FaceMeshFilter.fitPose(srcMx, srcMy, pose)
        var scale2 = FaceMeshFilter.viewScale2(pose)
        if (!(scale2 > 1e-8f) || !scale2.isFinite()) scale2 = faceWidth * faceHeight / CANON_FH_OVER_FW
        for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
            val a = indices[t * 3].toInt()
            val b = indices[t * 3 + 1].toInt()
            val c = indices[t * 3 + 2].toInt()
            val projected = 0.5f * ((srcMx[b] - srcMx[a]) * (srcMy[c] - srcMy[a]) -
                (srcMy[b] - srcMy[a]) * (srcMx[c] - srcMx[a])) * mirrorSign
            triVis[t] = smoothstep(0f, RHO_FADE, projected / (scale2 * TRI_AREA_3D[t]))
        }
        for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
            var sumW = 0f
            var sumV = 0f
            for (p in VERTEX_TRI_OFFSETS[i] until VERTEX_TRI_OFFSETS[i + 1]) {
                val t = VERTEX_TRI_INDICES[p].toInt()
                sumW += TRI_AREA_3D[t]
                sumV += TRI_AREA_3D[t] * triVis[t]
            }
            val v = if (sumW > 0f) sumV / sumW else 0f
            vis[i] = v * v
        }
        repeat(2) {
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val start = NEIGHBOR_OFFSETS[i]
                val end = NEIGHBOR_OFFSETS[i + 1]
                if (end > start) {
                    var sum = 0f
                    for (p in start until end) sum += vis[NEIGHBOR_INDICES[p].toInt()]
                    visScratch[i] = 0.5f * vis[i] + 0.5f * (sum / (end - start))
                } else {
                    visScratch[i] = vis[i]
                }
            }
            visScratch.copyInto(vis)
        }
        if (temporal) {
            if (temporalFresh) {
                vis.copyInto(visState)
            } else {
                val down = 1f - exp(-dtMs / VIS_TAU_DOWN_MS)
                val up = 1f - exp(-dtMs / VIS_TAU_UP_MS)
                for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                    val target = vis[i]
                    visState[i] += (if (target < visState[i]) down else up) * (target - visState[i])
                    vis[i] = visState[i]
                }
            }
        }

        // 面部参考中心与各解剖锚点（均在度量空间）。
        val cx = 0.25f * (srcMx[234] + srcMx[454] + srcMx[10] + srcMx[152])
        val cy = 0.25f * (srcMy[234] + srcMy[454] + srcMy[10] + srcMy[152])
        val fw = faceWidth
        val fh = faceHeight

        // 解剖右眼 33..133，上下睑 159/145，眉心 105
        val rEyeX = 0.25f * (srcMx[33] + srcMx[133] + srcMx[159] + srcMx[145])
        val rEyeY = 0.25f * (srcMy[33] + srcMy[133] + srcMy[159] + srcMy[145])
        val rEyeW = hypot(srcMx[33] - srcMx[133], srcMy[33] - srcMy[133]).coerceAtLeast(0.01f)
        val rEyeH = hypot(srcMx[159] - srcMx[145], srcMy[159] - srcMy[145]).coerceAtLeast(0.004f)
        val rBrowV = (srcMx[105] - rEyeX) * vx + (srcMy[105] - rEyeY) * vy

        // 解剖左眼 362..263，上下睑 386/374，眉心 334
        val lEyeX = 0.25f * (srcMx[362] + srcMx[263] + srcMx[386] + srcMx[374])
        val lEyeY = 0.25f * (srcMy[362] + srcMy[263] + srcMy[386] + srcMy[374])
        val lEyeW = hypot(srcMx[362] - srcMx[263], srcMy[362] - srcMy[263]).coerceAtLeast(0.01f)
        val lEyeH = hypot(srcMx[386] - srcMx[374], srcMy[386] - srcMy[374]).coerceAtLeast(0.004f)
        val lBrowV = (srcMx[334] - lEyeX) * vx + (srcMy[334] - lEyeY) * vy

        // 鼻尖 1、鼻翼 129/358、鼻根 168
        val noseTipX = srcMx[1]; val noseTipY = srcMy[1]
        val noseCenterX = 0.5f * (srcMx[129] + srcMx[358]); val noseCenterY = 0.5f * (srcMy[129] + srcMy[358])
        val noseHalfW = (0.5f * hypot(srcMx[358] - srcMx[129], srcMy[358] - srcMy[129])).coerceAtLeast(0.01f)
        val noseLenBase = hypot(srcMx[168] - noseTipX, srcMy[168] - noseTipY).coerceAtLeast(0.02f)

        // 嘴心与嘴角 61/291
        val mouthX = 0.5f * (srcMx[61] + srcMx[291])
        val mouthY = 0.5f * (srcMy[61] + srcMy[291])
        val mouthHalfW = (0.5f * hypot(srcMx[291] - srcMx[61], srcMy[291] - srcMy[61])).coerceAtLeast(0.01f)
        val mouthHalfH = max(0.5f * hypot(srcMx[0] - srcMx[17], srcMy[0] - srcMy[17]), 0.35f * mouthHalfW)

        // 调制后的有效强度。
        val faceMod = presence * steadyFace
        val eyeMod = presence * steadyEye
        val symFaceMod = faceMod * yawWeight
        val symEyeMod = eyeMod * yawWeight

        val sFaceSlim = params.faceSlim.coerceIn(0f, 1f) * faceMod
        val sVFace = params.vFace.coerceIn(0f, 1f) * faceMod
        val sFaceNarrow = params.faceNarrow.coerceIn(0f, 1f) * faceMod
        val sFaceSmall = params.faceSmall.coerceIn(0f, 1f) * symFaceMod
        val sCheekbone = params.cheekboneSlim.coerceIn(0f, 1f) * faceMod
        val sChin = params.chinShape.coerceIn(-1f, 1f) * symFaceMod
        val sForehead = params.foreheadShape.coerceIn(-1f, 1f) * symFaceMod
        val sCranial = params.cranialTop.coerceIn(0f, 1f) * symFaceMod
        val sEyeEnlarge = params.eyeEnlarge.coerceIn(0f, 1f) * eyeMod
        val sEyeDist = params.eyeDistance.coerceIn(-1f, 1f) * symEyeMod
        val sEyeCorner = params.eyeCorner.coerceIn(-1f, 1f) * eyeMod
        val sNoseSlim = params.noseSlim.coerceIn(0f, 1f) * symEyeMod
        val sNoseLen = params.noseLength.coerceIn(-1f, 1f) * symEyeMod
        val sMouthSize = params.mouthSize.coerceIn(-1f, 1f) * symEyeMod
        val sMouthSmile = params.mouthSmile.coerceIn(0f, 1f) * symEyeMod

        // 与 FaceMeshFrame 同源的无透视偏置屏幕偏航比：常规手持自拍（|signedYaw| <= 0.36，约 24° 内）保持双侧满权重 1.0，
        // 即使上游直接传入未过 FaceMeshFrame 的原始点集，>24°..42° 大侧脸远侧轮廓也会平滑渐出防过度推拉。
        val span132To361 = hypot(srcMx[361] - srcMx[132], srcMy[361] - srcMy[132]).coerceAtLeast(0.05f)
        val cheekMidX = 0.5f * (srcMx[132] + srcMx[361])
        val cheekMidY = 0.5f * (srcMy[132] + srcMy[361])
        val signedYaw = ((srcMx[1] - cheekMidX) * ux + (srcMy[1] - cheekMidY) * uy) / span132To361
        val poseRightFade = 1f - 0.82f * smoothstep(0.36f, 0.66f, signedYaw)
        val poseLeftFade = 1f - 0.82f * smoothstep(0.36f, 0.66f, -signedYaw)
        val effRightW = min(rightWeight, poseRightFade)
        val effLeftW = min(leftWeight, poseLeftFade)
        val rSideW = if (mirrorSign > 0f) effLeftW else effRightW
        val lSideW = if (mirrorSign > 0f) effRightW else effLeftW
        val maxDisp = 0.135f * max(fw, fh)
        val canonicalUv = FaceMeshTopology.CANONICAL_UV

        // 基于骨相正中矢状面（眉间 168、鼻底 2、下巴尖 152）与零偏置 C∞ Boltzmann SoftMax 提取屏幕左右真实 2D 外轮廓半宽 (wLeft, wRight)：
        // - 远侧 FACE_OVAL 折向脸颊后方时自动锁定正面前颊视平线外沿，彻底摆脱 3D 掠射法向 vis 14 倍跷跷板突变；
        // - 正脸自拍（|signedYaw| < 0.14）下对天生/广角偏心较宽一侧施加对称均衡补偿，偏头时 symExp 平滑归零、两侧按各自半宽等比例收缩，彻底根治「大小脸」。
        val lineMidX = 0.5f * (srcMx[10] + srcMx[152])
        val lineMidY = 0.5f * (srcMy[10] + srcMy[152])
        val midBoneX = 0.35f * srcMx[168] + 0.35f * srcMx[2] + 0.30f * srcMx[152]
        val midBoneY = 0.35f * srcMy[168] + 0.35f * srcMy[2] + 0.30f * srcMy[152]
        val halfH = max(0.5f * fh, 0.025f)
        val spanBeta = 64f / max(fw, 0.05f)
        val excessBeta = 48f / max(fw, 0.05f)
        val leftCandidates = if (mirrorSign > 0f) CHEEK_ANAT_RIGHT else CHEEK_ANAT_LEFT
        val rightCandidates = if (mirrorSign > 0f) CHEEK_ANAT_LEFT else CHEEK_ANAT_RIGHT
        val wLeft = max(softMaxSpan(leftCandidates, -1f, midBoneX, midBoneY, ux, uy, spanBeta), 0.10f * fw)
        val wRight = max(softMaxSpan(rightCandidates, 1f, midBoneX, midBoneY, ux, uy, spanBeta), 0.10f * fw)
        val wMean = max(0.5f * (wLeft + wRight), 0.16f * fw)
        val frontGate = 1f - smoothstep(0.05f, 0.14f, abs(signedYaw))
        val symExp = 0.85f * frontGate
        val lSym = (wLeft / wMean).pow(symExp).coerceIn(0.80f, 1.25f)
        val rSym = (wRight / wMean).pow(symExp).coerceIn(0.80f, 1.25f)
        val contourActive = sFaceSlim > 1e-4f || sVFace > 1e-4f || sFaceNarrow > 1e-4f || sCheekbone > 1e-4f

        // 2D 保面积剪切平台微分同胚轮廓场：在度量平面 (du, dv) 上对脸内 (0..467) 与内层随动过渡环 (468..503) 一致求值；
        // 外颊至下颌轮廓带（xi ∈ [0.60, 1.35]）处于 eta == 1 纯竖向剪切平台（∂(dU)/∂(du) == 0，det J == 1），
        // 既不在斜向下颌边产生二次弯曲压扁掠射细三角，也彻底杜绝正面内颊点跨越背折 FACE_OVAL 翻出未剔除重影三角（从 2835 px² 降至 ≈ 0）。
        fun evalContour2D(px: Float, py: Float): Pair<Float, Float> {
            if (!contourActive) return 0f to 0f
            val du = (px - midBoneX) * ux + (py - midBoneY) * uy
            val dv = ((px - lineMidX) * vx + (py - lineMidY) * vy) / halfH
            val sideBlend = smoothstep(-0.14f * wMean, 0.14f * wMean, du)
            val wSide = wLeft + (wRight - wLeft) * sideBlend
            val sideW = effLeftW + (effRightW - effLeftW) * sideBlend
            val symW = lSym + (rSym - lSym) * sideBlend
            val jawTaper = 1f - 0.36f * smoothstep(-0.35f, -0.92f, dv)
            val wLocal = max(wSide * jawTaper, 0.065f * fw)
            val xi = abs(du) / wLocal
            if (xi >= 2.15f) return 0f to 0f

            val innerGate = smoothstep(0.32f, 0.56f, xi)
            val outerGate = 1f - smoothstep(1.35f, 2.10f, xi)
            val vertGuard = 1f - smoothstep(0.95f, 1.32f, -dv)
            val eta = innerGate * outerGate
            if (eta <= 1e-5f) return 0f to 0f

            var frac = 0f
            var liftV = 0f
            if (sFaceSlim > 0f) {
                val slimVert = smoothstep(-1.18f, -0.76f, dv) * (1f - smoothstep(0.15f, 0.42f, dv))
                frac += sFaceSlim * 0.068f * slimVert
            }
            if (sVFace > 0f) {
                val vVert = gaussian(dv, -0.58f, 0.48f) * vertGuard
                frac += sVFace * 0.068f * vVert
                liftV += sVFace * 0.012f * fh * sideW * eta * vVert
            }
            if (sFaceNarrow > 0f) {
                frac += sFaceNarrow * 0.065f * gaussian(dv, -0.05f, 0.68f)
            }
            if (sCheekbone > 0f) {
                frac += sCheekbone * 0.070f * gaussian(dv, 0.06f, 0.36f) * smoothstep(-0.55f, -0.20f, dv) * vertGuard
            }
            frac *= symW * sideW * eta
            val maxFrac = 0.115f * symW
            if (frac > 0f && maxFrac > 1e-5f) {
                frac = (frac * maxFrac) / hypot(frac, maxFrac)
            }
            val sgn = 2f * sideBlend - 1f
            val cDu = -sgn * wSide * frac
            return (cDu * ux + liftV * vx) to (cDu * uy + liftV * vy)
        }

        // 2. 逐顶点累积五官解剖微整与 2D 连续轮廓形变场（0..467）。
        if (faceMod > 0f || eyeMod > 0f) {
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val px = srcMx[i]; val py = srcMy[i]
                val vVis = vis[i]
                var intDx = 0f
                var intDy = 0f
                if (vVis > 1e-4f) {
                    val rx = px - cx; val ry = py - cy
                    val du = rx * ux + ry * uy
                    val dv = rx * vx + ry * vy
                    val nu = (canonicalUv[i * 2] - 0.5f) * 2f
                    val screenNu = nu * mirrorSign
                    val nv = (0.5f - canonicalUv[i * 2 + 1]) * 2f
                    val sideT = ((screenNu + 0.15f) / 0.30f).coerceIn(0f, 1f)
                    val sideW = effLeftW + (effRightW - effLeftW) * sideT
                    val absNu = abs(screenNu)
                    var dU = 0f
                    var dV = 0f

                    // (4) 小脸 faceSmall：整张脸相对面中各向同性微缩。
                    if (sFaceSmall > 0f) {
                        val scale = sFaceSmall * 0.060f
                        dU -= du * scale * sideW
                        dV -= dv * scale
                    }

                    // (6) 下巴 chinShape（双向）：正值拉长下巴、负值缩短下巴；锚定真实下巴尖 152。
                    if (abs(sChin) > 1e-4f) {
                        val w = radialFalloff(px, py, srcMx[152], srcMy[152], 0.34f * fh)
                        if (w > 0f) {
                            dV -= sChin * 0.095f * fh * w
                            if (sChin > 0f) dU -= du * (sChin * 0.065f * w)
                        }
                    }

                    // (7) 额头/发际线 foreheadShape（双向）：正值抬高发际线、负值拉低发际线；锚定真实额头 10，保护眉眼。
                    if (abs(sForehead) > 1e-4f && nv > 0.32f) {
                        val vert = smoothstep(0.32f, 0.78f, nv)
                        val lat = 1f - smoothstep(0.70f, 1.05f, absNu) * 0.35f
                        dV += sForehead * 0.092f * fh * vert * lat
                    }

                    // (7') 高颅顶发际线防拉长（cranialTop）：微收上发际线 0.6%，防止外蓬颅顶发冠时把裸额头皮肤带高。
                    if (sCranial > 1e-4f && nv > 0.58f) {
                        val tuck = smoothstep(0.58f, 0.92f, nv) * (1f - smoothstep(0.65f, 1.0f, absNu) * 0.5f)
                        dV -= sCranial * 0.006f * fh * tuck
                    }

                    // (8) 大眼 eyeEnlarge：按真实上下眼睑各向异性撑开，上边界接近眉毛时平滑衰减避免顶眉。
                    if (sEyeEnlarge > 0f) {
                        val re = eyeEnlargeDelta(px, py, rEyeX, rEyeY, rEyeW, rEyeH, rBrowV, ux, uy, vx, vy, sEyeEnlarge * rSideW)
                        dU += re.first; dV += re.second
                        val le = eyeEnlargeDelta(px, py, lEyeX, lEyeY, lEyeW, lEyeH, lBrowV, ux, uy, vx, vy, sEyeEnlarge * lSideW)
                        dU += le.first; dV += le.second
                    }

                    // (9) 眼距 eyeDistance（双向）：正值拉宽、负值拉近；分别平移左右整眼与眼眶区域。
                    if (abs(sEyeDist) > 1e-4f) {
                        val wr = radialFalloff(px, py, rEyeX, rEyeY, 1.25f * rEyeW)
                        val wl = radialFalloff(px, py, lEyeX, lEyeY, 1.25f * lEyeW)
                        if (wr > 0f) dU -= mirrorSign * sEyeDist * 0.18f * rEyeW * wr
                        if (wl > 0f) dU += mirrorSign * sEyeDist * 0.18f * lEyeW * wl
                    }

                    // (10) 眼角 eyeCorner（双向）：正值上扬外眼角（猫眼/狐狸眼），负值下垂（无辜眼）。
                    if (abs(sEyeCorner) > 1e-4f) {
                        val wr = radialFalloff(px, py, srcMx[33], srcMy[33], 0.85f * rEyeW)
                        val wl = radialFalloff(px, py, srcMx[263], srcMy[263], 0.85f * lEyeW)
                        if (wr > 0f) dV += sEyeCorner * 0.18f * rEyeW * wr * rSideW
                        if (wl > 0f) dV += sEyeCorner * 0.18f * lEyeW * wl * lSideW
                    }

                    // (11) 瘦鼻 noseSlim：围绕真实鼻翼 129/358 向鼻梁中线精准内收。
                    if (sNoseSlim > 0f) {
                        val dnx = px - noseCenterX; val dny = py - noseCenterY
                        val nDu = dnx * ux + dny * uy
                        val nDv = dnx * vx + dny * vy
                        val nr = hypot(nDu / (1.85f * noseHalfW), nDv / (1.45f * noseHalfW))
                        if (nr < 1f) {
                            val w = (1f - nr * nr).let { it * it }
                            dU -= nDu * (sNoseSlim * 0.28f * w)
                        }
                    }

                    // (12) 长鼻/缩鼻 noseLength（双向）：正值微拉长鼻尖下移、负值微缩短鼻尖上提，保护上唇。
                    if (abs(sNoseLen) > 1e-4f) {
                        val dnx = px - noseTipX; val dny = py - noseTipY
                        val nDu = dnx * ux + dny * uy
                        val nDv = dnx * vx + dny * vy
                        val nr = hypot(nDu / (1.65f * noseHalfW), nDv / (0.85f * noseLenBase))
                        if (nr < 1f) {
                            val w = (1f - nr * nr).let { it * it }
                            dV -= sNoseLen * 0.15f * noseLenBase * w
                        }
                    }

                    // (13) 嘴型大小 mouthSize（双向）：正值丰满放大、负值秀气缩小。
                    if (abs(sMouthSize) > 1e-4f) {
                        val dmx = px - mouthX; val dmy = py - mouthY
                        val mDu = dmx * ux + dmy * uy
                        val mDv = dmx * vx + dmy * vy
                        val mr = hypot(mDu / (1.55f * mouthHalfW), mDv / (1.95f * mouthHalfH))
                        if (mr < 1f) {
                            val w = (1f - mr * mr).let { it * it }
                            dU += mDu * (sMouthSize * 0.22f * w)
                            dV += mDv * (sMouthSize * 0.18f * w)
                        }
                    }

                    // (14) 微笑唇 mouthSmile：左右嘴角 61/291 自然上扬微展，避免牵动脸颊僵硬。
                    if (sMouthSmile > 0f) {
                        val wr = radialFalloff(px, py, srcMx[61], srcMy[61], 0.72f * mouthHalfW)
                        val wl = radialFalloff(px, py, srcMx[291], srcMy[291], 0.72f * mouthHalfW)
                        if (wr > 0f) {
                            dV += sMouthSmile * 0.16f * mouthHalfW * wr
                            dU -= mirrorSign * sMouthSmile * 0.045f * mouthHalfW * wr
                        }
                        if (wl > 0f) {
                            dV += sMouthSmile * 0.16f * mouthHalfW * wl
                            dU += mirrorSign * sMouthSmile * 0.045f * mouthHalfW * wl
                        }
                    }

                    dU *= vVis
                    dV *= vVis
                    intDx = dU * ux + dV * vx
                    intDy = dU * uy + dV * vy
                }
                interiorMx[i] = intDx
                interiorMy[i] = intDy
            }

            // 仅对脸内五官微整做一阶拓扑拉普拉斯平滑；2D 轮廓场本身已是屏幕空间 C∞ 映射，不跨 3D 背折视平线混入耳后拓扑邻域以防剪切细边。
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val start = NEIGHBOR_OFFSETS[i]
                val end = NEIGHBOR_OFFSETS[i + 1]
                if (end > start) {
                    var sx = 0f
                    var sy = 0f
                    for (p in start until end) {
                        val nb = NEIGHBOR_INDICES[p].toInt()
                        sx += interiorMx[nb]
                        sy += interiorMy[nb]
                    }
                    val inv = 1f / (end - start)
                    smoothMx[i] = 0.80f * interiorMx[i] + 0.20f * (sx * inv)
                    smoothMy[i] = 0.80f * interiorMy[i] + 0.20f * (sy * inv)
                } else {
                    smoothMx[i] = interiorMx[i]
                    smoothMy[i] = interiorMy[i]
                }
            }
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                interiorMx[i] = smoothMx[i]
                interiorMy[i] = smoothMy[i]
                val (conDx, conDy) = evalContour2D(srcMx[i], srcMy[i])
                var totDx = interiorMx[i] + conDx
                var totDy = interiorMy[i] + conDy
                val mag = hypot(totDx, totDy)
                if (mag > 1e-6f && maxDisp > 1e-5f) {
                    val capScale = maxDisp / hypot(mag, maxDisp)
                    totDx *= capScale
                    totDy *= capScale
                }
                dispMx[i] = totDx
                dispMy[i] = totDy
            }
        }

        // 3. 计算视平线包络双环 C1 紧支集缓冲环：
        //    - 内层随动环（468..503）经 SoftMax 自动跨出远侧背折视平线并与脸内共享同一 2D 连续形变场，包裹真实皮肉边缘与微小跟脸滞后整体内移，消除轮廓残影；
        //    - 外层锚定环（504..539）与 8 个屏幕外锚点（540..547）严格恒零位移，截断背景门框弯曲。
        val oval = FaceMeshTopology.FACE_OVAL
        val padMidBase = max(0.055f * fw, 0.048f * fh)
        val padOuterBase = max(0.185f * fw, 0.165f * fh)
        for (k in 0 until FaceMeshTopology.OVAL_VERTEX_COUNT) {
            val oi = oval[k]
            val ox = srcMx[oi]; val oy = srcMy[oi]
            val cNu = (canonicalUv[oi * 2] - 0.5f) * 2f * mirrorSign
            val cNv = (0.5f - canonicalUv[oi * 2 + 1]) * 2f
            val cLen = hypot(cNu, cNv).coerceAtLeast(1e-4f)
            val cxDir = (cNu * ux + cNv * vx) / cLen
            val cyDir = (cNu * uy + cNv * vy) / cLen
            val dx = ox - cx; val dy = oy - cy
            val len = hypot(dx, dy)
            val radX = if (len > 1e-4f) dx / len else cxDir
            val radY = if (len > 1e-4f) dy / len else cyDir
            val align = (radX * cxDir + radY * cyDir).coerceIn(0f, 1f)
            val wRad = 0.65f * align
            val mx = cxDir * (1f - wRad) + radX * wRad
            val my = cyDir * (1f - wRad) + radY * wRad
            val mNorm = hypot(mx, my).coerceAtLeast(1e-4f)
            val rx = mx / mNorm; val ry = my / mNorm
            val silExcess = min(softMaxOvalExcess(k, ox, oy, rx, ry, excessBeta), 0.070f * fw)
            val outwardDisp = max(0f, dispMx[oi] * rx + dispMy[oi] * ry)
            val crownW = if (sCranial > 1e-4f) smoothstep(-0.10f, 0.68f, cNv) else 0f
            val templeW = if (sCranial > 1e-4f) smoothstep(0.02f, 0.44f, cNv) * (1f - smoothstep(0.72f, 0.98f, cNv)) else 0f
            val cranialSpan = sCranial * crownW * 0.095f * fh
            val padMid = padMidBase + silExcess + outwardDisp * 0.70f + cranialSpan
            val midX = ox + rx * padMid
            val midY = oy + ry * padMid
            val mi = FaceMeshTopology.MID_BUFFER_OFFSET + k
            srcMx[mi] = midX; srcMy[mi] = midY
            dstMx[mi] = midX; dstMy[mi] = midY
            if (faceMod > 0f || eyeMod > 0f) {
                val (mcx, mcy) = evalContour2D(midX, midY)
                var mdx = mcx + interiorMx[oi] * MID_INTERIOR_CARRY
                var mdy = mcy + interiorMy[oi] * MID_INTERIOR_CARRY
                if (sCranial > 1e-4f && (crownW > 0f || templeW > 0f)) {
                    val liftV = sCranial * crownW * 0.072f * fh
                    val sideSgn = smoothstep(-0.18f, 0.18f, cNu) * 2f - 1f
                    val fluffU = sCranial * templeW * 0.026f * fw * sideSgn
                    mdx += fluffU * ux + liftV * vx
                    mdy += fluffU * uy + liftV * vy
                }
                dispMx[mi] = mdx
                dispMy[mi] = mdy
            }

            val padOuter = padOuterBase + silExcess * 0.85f + outwardDisp * 1.50f + sCranial * 0.185f * fh
            val bx = ox + rx * padOuter
            val by = oy + ry * padOuter
            val bi = FaceMeshTopology.BUFFER_OFFSET + k
            srcMx[bi] = bx; srcMy[bi] = by
            dstMx[bi] = bx; dstMy[bi] = by
        }
        val outerR = max(3.2f, max(fw, fh) * 5.0f)
        val outerDirs = FaceMeshTopology.OUTER_DIRECTIONS
        for (j in 0 until FaceMeshTopology.OUTER_VERTEX_COUNT) {
            // 横向锚点乘 mirrorSign，使 8 个外框点绕序与 FACE_OVAL 严格同向，杜绝跨脸对角覆盖。
            val du = outerDirs[j * 2] * mirrorSign * outerR
            val dv = outerDirs[j * 2 + 1] * outerR
            val fx = cx + du * ux + dv * vx
            val fy = cy + du * uy + dv * vy
            val fi = FaceMeshTopology.OUTER_OFFSET + j
            srcMx[fi] = fx; srcMy[fi] = fy
            dstMx[fi] = fx; dstMy[fi] = fy
        }

        // 4. 应用位移 + 连续障碍函数防翻转守卫：固定 RELAX_ITER 轮（不提前退出，输入到输出的映射连续），
        //    只约束原图真正正朝向的三角形（形状权重 w 随「2 × 面积 / 最长边²」从 0 平滑升到 1，背面与退化三角形为 0），
        //    面积比 r 低于 RELAX_R1 起平滑收缩，邻域平滑回退，绝不产生单点归零锯齿裂缝。
        for (t in 0 until FaceMeshTopology.DEFORMABLE_TRIANGLE_COUNT) {
            val a = indices[t * 3].toInt()
            val b = indices[t * 3 + 1].toInt()
            val c = indices[t * 3 + 2].toInt()
            val ex1 = srcMx[b] - srcMx[a]; val ey1 = srcMy[b] - srcMy[a]
            val ex2 = srcMx[c] - srcMx[a]; val ey2 = srcMy[c] - srcMy[a]
            val ex3 = srcMx[c] - srcMx[b]; val ey3 = srcMy[c] - srcMy[b]
            val origArea = ex1 * ey2 - ey1 * ex2
            val maxEdge2 = max(ex1 * ex1 + ey1 * ey1, max(ex2 * ex2 + ey2 * ey2, ex3 * ex3 + ey3 * ey3))
            triOrig[t] = origArea
            triFront[t] = if (origArea * mirrorSign > 0f && maxEdge2 > 0f) {
                smoothstep(0f, FRONT_W_SHAPE, origArea * mirrorSign / maxEdge2)
            } else 0f
        }
        // 时间平滑放在守卫之前：上一帧的松弛系数按 RELAX_TAU_UP_MS 向 1 恢复后作为本帧的起点（上限），守卫与最后一道安全
        // 只会再收小（变小立即生效、变大约 120 ms 恢复），最终用于绘制的就是守卫校验过的那一组系数。
        // 不能在守卫之后再逐顶点改系数：相邻顶点回升不同步，细长的正面三角形（嘴角、鼻翼、唇缘）会翻面。
        if (temporal && !temporalFresh) {
            val up = 1f - exp(-dtMs / RELAX_TAU_UP_MS)
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) relax[i] = relaxState[i] + up * (1f - relaxState[i])
        } else {
            relax.fill(1f)
        }
        applyDisplacement()
        for (iter in 0 until RELAX_ITER) {
            relaxFactor.fill(1f)
            for (t in 0 until FaceMeshTopology.DEFORMABLE_TRIANGLE_COUNT) {
                val w = triFront[t]
                if (w <= 0f) continue
                val r = areaRatio(t, indices)
                val h = RELAX_H_FLOOR + (1f - RELAX_H_FLOOR) * smoothstep(RELAX_R0, RELAX_R1, r)
                val f = 1f - w * (1f - h)
                if (f >= 1f) continue
                limitVertices(t, indices, f)
            }
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) relax[i] *= relaxFactor[i]
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val start = NEIGHBOR_OFFSETS[i]
                val end = NEIGHBOR_OFFSETS[i + 1]
                if (end > start) {
                    var minNb = relax[i]
                    var sumNb = 0f
                    for (p in start until end) {
                        val rv = relax[NEIGHBOR_INDICES[p].toInt()]
                        if (rv < minNb) minNb = rv
                        sumNb += rv
                    }
                    val avgNb = sumNb / (end - start)
                    relaxScratch[i] = min(relax[i], 0.5f * (relax[i] + min(avgNb, minNb + 0.25f)))
                } else {
                    relaxScratch[i] = relax[i]
                }
            }
            relaxScratch.copyInto(relax)
            applyDisplacement()
        }
        // 最后一道安全（连续）：仍被压得很扁的正面三角形，三个顶点的松弛系数平滑夹到不超过 1 − SAFETY_DEPTH。
        relaxFactor.fill(1f)
        for (t in 0 until FaceMeshTopology.DEFORMABLE_TRIANGLE_COUNT) {
            val w = triFront[t]
            if (w <= 0f) continue
            val cap = 1f - SAFETY_DEPTH * smoothstep(SAFETY_W0, SAFETY_W1, w) *
                (1f - smoothstep(SAFETY_R0, SAFETY_R1, areaRatio(t, indices)))
            if (cap < 1f) limitVertices(t, indices, cap)
        }
        for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) relax[i] *= relaxFactor[i]
        if (temporal) {
            relax.copyInto(relaxState)
            temporalValid = true
        }
        applyDisplacement()

        // 5. 打包 512 顶点 GPU 缓冲与 468 点变形纹理坐标。
        for (i in 0 until FaceMeshTopology.TOTAL_VERTEX_COUNT) {
            val sx = srcMx[i]
            val sy = srcMy[i] * aspect
            val dx = dstMx[i]
            val dy = dstMy[i] * aspect
            val k = i * 4
            vertexData[k] = dx * 2f - 1f
            vertexData[k + 1] = dy * 2f - 1f
            vertexData[k + 2] = sx
            vertexData[k + 3] = sy
            if (i < FaceMeshTopology.FACE_VERTEX_COUNT) {
                warpedLandmarks[i * 2] = dx
                warpedLandmarks[i * 2 + 1] = dy
            }
        }

        // 6. 同步变换 10 个虹膜关键点（468..472 右眼、473..477 左眼），与眼眶变形精准对齐供美瞳使用。
        transformIris(landmarks, 468, 33, 133, 159, 145, aspect)
        transformIris(landmarks, 473, 362, 263, 386, 374, aspect)
        return true
    }

    /** 第 [t] 个三角形变形后与原图的有向面积比（镜像无关）。 */
    private fun areaRatio(t: Int, indices: ShortArray): Float {
        val a = indices[t * 3].toInt()
        val b = indices[t * 3 + 1].toInt()
        val c = indices[t * 3 + 2].toInt()
        val newArea = (dstMx[b] - dstMx[a]) * (dstMy[c] - dstMy[a]) - (dstMy[b] - dstMy[a]) * (dstMx[c] - dstMx[a])
        return newArea / triOrig[t]
    }

    /** 第 [t] 个三角形的可动顶点（内层过渡环映射回对应脸廓点，外缓冲环与外框恒不动）按压缩度平滑向三点最小松弛系数对齐并收缩。 */
    private fun limitVertices(t: Int, indices: ShortArray, f: Float) {
        var minR = 1f
        for (k in 0 until 3) {
            val v = indices[t * 3 + k].toInt()
            val target = when {
                v < FaceMeshTopology.FACE_VERTEX_COUNT -> v
                v < FaceMeshTopology.BUFFER_OFFSET -> FaceMeshTopology.FACE_OVAL[v - FaceMeshTopology.MID_BUFFER_OFFSET]
                else -> -1
            }
            if (target >= 0 && relax[target] < minR) minR = relax[target]
        }
        val eq = ((1f - f) * 2.5f).coerceIn(0f, 1f)
        for (k in 0 until 3) {
            val v = indices[t * 3 + k].toInt()
            val target = when {
                v < FaceMeshTopology.FACE_VERTEX_COUNT -> v
                v < FaceMeshTopology.BUFFER_OFFSET -> FaceMeshTopology.FACE_OVAL[v - FaceMeshTopology.MID_BUFFER_OFFSET]
                else -> -1
            }
            if (target >= 0) {
                val cur = relax[target]
                val effF = if (cur > 1e-6f) f * (cur + eq * (minR - cur)) / cur else f
                if (effF < relaxFactor[target]) relaxFactor[target] = effF
            }
        }
    }

    private fun applyDisplacement() {
        for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
            val r = relax[i]
            dstMx[i] = srcMx[i] + dispMx[i] * r
            dstMy[i] = srcMy[i] + dispMy[i] * r
        }
        for (k in 0 until FaceMeshTopology.OVAL_VERTEX_COUNT) {
            val oi = FaceMeshTopology.FACE_OVAL[k]
            val mi = FaceMeshTopology.MID_BUFFER_OFFSET + k
            val r = relax[oi]
            dstMx[mi] = srcMx[mi] + dispMx[mi] * r
            dstMy[mi] = srcMy[mi] + dispMy[mi] * r
        }
    }

    /** C∞ Boltzmann 加权 SoftMax 提取单侧脸颊/下颌点集沿 [signDir]·(ux, uy) 的无偏真实 2D 视平线半宽（零分配）。 */
    private fun softMaxSpan(
        candidates: ShortArray,
        signDir: Float,
        midX: Float,
        midY: Float,
        ux: Float,
        uy: Float,
        beta: Float,
    ): Float {
        if (candidates.isEmpty()) return 0f
        var maxVal = 0f
        for (idx in candidates) {
            val i = idx.toInt()
            val v = max(0f, signDir * ((srcMx[i] - midX) * ux + (srcMy[i] - midY) * uy))
            if (v > maxVal) maxVal = v
        }
        var sumWeighted = 0f
        var sumExp = 0f
        for (idx in candidates) {
            val i = idx.toInt()
            val v = max(0f, signDir * ((srcMx[i] - midX) * ux + (srcMy[i] - midY) * uy))
            val w = exp(beta * (v - maxVal))
            sumWeighted += v * w
            sumExp += w
        }
        return if (sumExp > 0f) sumWeighted / sumExp else 0f
    }

    /** C∞ Boltzmann 加权 SoftMax 求第 [ovalSlot] 个轮廓点二环邻域超出该轮廓点的正前颊视平线外凸量（零分配）。 */
    private fun softMaxOvalExcess(
        ovalSlot: Int,
        ox: Float,
        oy: Float,
        rx: Float,
        ry: Float,
        beta: Float,
    ): Float {
        val start = OVAL_NB2_OFFSETS[ovalSlot]
        val end = OVAL_NB2_OFFSETS[ovalSlot + 1]
        if (end <= start) return 0f
        var maxVal = 0f
        for (p in start until end) {
            val v = OVAL_NB2_INDICES[p].toInt()
            val ex = max(0f, (srcMx[v] - ox) * rx + (srcMy[v] - oy) * ry)
            if (ex > maxVal) maxVal = ex
        }
        var sumWeighted = 0f
        var sumExp = 0f
        for (p in start until end) {
            val v = OVAL_NB2_INDICES[p].toInt()
            val ex = max(0f, (srcMx[v] - ox) * rx + (srcMy[v] - oy) * ry)
            val w = exp(beta * (ex - maxVal))
            sumWeighted += ex * w
            sumExp += w
        }
        return if (sumExp > 0f) sumWeighted / sumExp else 0f
    }

    /** 基于眼睛四锚点变形前后的中心平移与尺度缩放，仿射更新对应 5 个虹膜点。 */
    private fun transformIris(
        src: FloatArray,
        baseIdx: Int,
        outer: Int,
        inner: Int,
        top: Int,
        bottom: Int,
        aspect: Float,
    ) {
        val oldCx = 0.25f * (src[outer * 2] + src[inner * 2] + src[top * 2] + src[bottom * 2])
        val oldCy = 0.25f * (src[outer * 2 + 1] + src[inner * 2 + 1] + src[top * 2 + 1] + src[bottom * 2 + 1])
        val newCx = 0.25f * (warpedLandmarks[outer * 2] + warpedLandmarks[inner * 2] +
            warpedLandmarks[top * 2] + warpedLandmarks[bottom * 2])
        val newCy = 0.25f * (warpedLandmarks[outer * 2 + 1] + warpedLandmarks[inner * 2 + 1] +
            warpedLandmarks[top * 2 + 1] + warpedLandmarks[bottom * 2 + 1])
        val oldW = hypot(src[outer * 2] - src[inner * 2], (src[outer * 2 + 1] - src[inner * 2 + 1]) / aspect).coerceAtLeast(1e-4f)
        val newW = hypot(warpedLandmarks[outer * 2] - warpedLandmarks[inner * 2],
            (warpedLandmarks[outer * 2 + 1] - warpedLandmarks[inner * 2 + 1]) / aspect).coerceAtLeast(1e-4f)
        val scale = (newW / oldW).coerceIn(0.75f, 1.35f)
        for (k in 0 until 5) {
            val idx = (baseIdx + k) * 2
            warpedLandmarks[idx] = newCx + (src[idx] - oldCx) * scale
            warpedLandmarks[idx + 1] = newCy + (src[idx + 1] - oldCy) * scale
        }
    }

    /** 真实眼睑各向异性大眼位移：竖向撑开略大于横向，上边界接近眉毛时平滑衰减，绝不顶眉。 */
    private fun eyeEnlargeDelta(
        px: Float,
        py: Float,
        eyeX: Float,
        eyeY: Float,
        eyeW: Float,
        eyeH: Float,
        browDistV: Float,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
        strength: Float,
    ): Pair<Float, Float> {
        if (strength <= 0f) return 0f to 0f
        val dx = px - eyeX; val dy = py - eyeY
        val eDu = dx * ux + dy * uy
        val eDv = dx * vx + dy * vy
        val rx = 1.05f * eyeW
        val ry = max(0.85f * eyeW, 1.6f * eyeH)
        val r = hypot(eDu / rx, eDv / ry)
        if (r >= 1f) return 0f to 0f
        var w = (1f - r * r).let { it * it }
        if (eDv > 0f && browDistV > 0.005f) {
            w *= (1f - smoothstep(0.50f * browDistV, 0.92f * browDistV, eDv))
        }
        val gainU = strength * 0.28f * w
        val gainV = strength * 0.36f * w
        return (eDu * gainU) to (eDv * gainV)
    }

    companion object {
        /** 可见度：ρ（法线与视线夹角余弦）从 0 到它平滑升到 1，再平方压平斜率（正脸轮廓约 0.3–0.6，背面为 0）。 */
        const val RHO_FADE = 0.15f
        /** 内层随动缓冲环承接下巴/发际线等纵向五官微整的平滑比例（横向轮廓微整直接由同一 2D 连续场求值）。 */
        const val MID_INTERIOR_CARRY = 0.55f
        /** 规范脸高 / 脸宽（姿态拟合失败时的回退投影尺度用）。 */
        const val CANON_FH_OVER_FW = 1.153f
        /** 可见度时间平滑：下降、上升的时间常数。 */
        const val VIS_TAU_DOWN_MS = 40f
        const val VIS_TAU_UP_MS = 120f
        /** 防翻转障碍函数：固定轮数；每轮收缩因子 h(r) = RELAX_H_FLOOR + (1 − RELAX_H_FLOOR)·smoothstep(RELAX_R0, RELAX_R1, r)。 */
        const val RELAX_ITER = 8
        const val RELAX_H_FLOOR = 0.65f
        const val RELAX_R0 = 0.08f
        const val RELAX_R1 = 0.30f
        /**
         * 正面形状权重：「2 × 面积 / 最长边²」从 0 到它平滑升到 1（正三角形约 0.87）。只让近乎退化、正在过零的三角形
         * 渐入约束（连续），规范网格里本来就细长的正面三角形（眼睑、唇缘约 0.005–0.02）照样满约束，保证不翻面。
         */
        const val FRONT_W_SHAPE = 0.002f
        /** 最后一道安全：形状权重在 [SAFETY_W0, SAFETY_W1]、面积比在 [SAFETY_R0, SAFETY_R1] 之间平滑过渡，最深夹到 1 − SAFETY_DEPTH。 */
        const val SAFETY_W0 = 0.3f
        const val SAFETY_W1 = 0.7f
        const val SAFETY_R0 = 0f
        const val SAFETY_R1 = 0.04f
        const val SAFETY_DEPTH = 0.5f
        /** 松弛系数变大时的恢复时间常数。 */
        const val RELAX_TAU_UP_MS = 120f
        /** 距上次 deform 超过这么久，时间平滑状态重新初始化。 */
        const val TEMPORAL_RESET_MS = 200L

        // 类加载期一次性构建 468 顶点压缩一环邻接表、轮廓二环邻域表与左右脸颊视平线候选集（运行期零分配）。
        private val NEIGHBOR_OFFSETS: IntArray
        private val NEIGHBOR_INDICES: ShortArray
        private val OVAL_NB2_OFFSETS: IntArray
        private val OVAL_NB2_INDICES: ShortArray
        private val CHEEK_ANAT_RIGHT: ShortArray
        private val CHEEK_ANAT_LEFT: ShortArray
        // 顶点 → 入射规范三角形（CSR），可见度按面积加权到顶点用。
        private val VERTEX_TRI_OFFSETS: IntArray
        private val VERTEX_TRI_INDICES: ShortArray
        /** 898 个规范三角形的 3D 面积（单位：规范脸宽²）。 */
        private val TRI_AREA_3D: FloatArray

        init {
            val adj = Array(FaceMeshTopology.FACE_VERTEX_COUNT) { HashSet<Int>(8) }
            val tris = FaceMeshTopology.CANONICAL_INDICES
            for (i in tris.indices step 3) {
                val a = tris[i].toInt()
                val b = tris[i + 1].toInt()
                val c = tris[i + 2].toInt()
                adj[a].add(b); adj[a].add(c)
                adj[b].add(a); adj[b].add(c)
                adj[c].add(a); adj[c].add(b)
            }
            val offsets = IntArray(FaceMeshTopology.FACE_VERTEX_COUNT + 1)
            var total = 0
            for (v in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                offsets[v] = total
                total += adj[v].size
            }
            offsets[FaceMeshTopology.FACE_VERTEX_COUNT] = total
            val flat = ShortArray(total)
            var pos = 0
            for (v in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                for (nb in adj[v]) flat[pos++] = nb.toShort()
            }
            NEIGHBOR_OFFSETS = offsets
            NEIGHBOR_INDICES = flat

            val oval = FaceMeshTopology.FACE_OVAL
            val ovalNb2 = Array(FaceMeshTopology.OVAL_VERTEX_COUNT) { k ->
                val oi = oval[k]
                val s = LinkedHashSet<Int>(24)
                s.add(oi)
                for (n1 in adj[oi]) {
                    s.add(n1)
                    s.addAll(adj[n1])
                }
                s
            }
            val nb2Offsets = IntArray(FaceMeshTopology.OVAL_VERTEX_COUNT + 1)
            var nb2Total = 0
            for (k in 0 until FaceMeshTopology.OVAL_VERTEX_COUNT) {
                nb2Offsets[k] = nb2Total
                nb2Total += ovalNb2[k].size
            }
            nb2Offsets[FaceMeshTopology.OVAL_VERTEX_COUNT] = nb2Total
            val nb2Flat = ShortArray(nb2Total)
            var nb2Pos = 0
            for (k in 0 until FaceMeshTopology.OVAL_VERTEX_COUNT) {
                for (v in ovalNb2[k]) nb2Flat[nb2Pos++] = v.toShort()
            }
            OVAL_NB2_OFFSETS = nb2Offsets
            OVAL_NB2_INDICES = nb2Flat

            val canonUv = FaceMeshTopology.CANONICAL_UV
            val rightList = ArrayList<Short>(96)
            val leftList = ArrayList<Short>(96)
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val nu = (canonUv[i * 2] - 0.5f) * 2f
                val nv = (0.5f - canonUv[i * 2 + 1]) * 2f
                if (nv in -0.72f..-0.04f) {
                    if (nu < -0.35f) rightList.add(i.toShort())
                    else if (nu > 0.35f) leftList.add(i.toShort())
                }
            }
            CHEEK_ANAT_RIGHT = rightList.toShortArray()
            CHEEK_ANAT_LEFT = leftList.toShortArray()

            val xyz = FaceMeshTopology.CANONICAL_XYZ
            val fw = FaceMeshFilter.CANONICAL_FACE_WIDTH
            val triCount = FaceMeshTopology.CANONICAL_TRIANGLE_COUNT
            TRI_AREA_3D = FloatArray(triCount) { t ->
                val a = tris[t * 3].toInt(); val b = tris[t * 3 + 1].toInt(); val c = tris[t * 3 + 2].toInt()
                val e1x = (xyz[b * 3] - xyz[a * 3]) / fw; val e1y = (xyz[b * 3 + 1] - xyz[a * 3 + 1]) / fw; val e1z = (xyz[b * 3 + 2] - xyz[a * 3 + 2]) / fw
                val e2x = (xyz[c * 3] - xyz[a * 3]) / fw; val e2y = (xyz[c * 3 + 1] - xyz[a * 3 + 1]) / fw; val e2z = (xyz[c * 3 + 2] - xyz[a * 3 + 2]) / fw
                val cx = e1y * e2z - e1z * e2y; val cy = e1z * e2x - e1x * e2z; val cz = e1x * e2y - e1y * e2x
                (0.5f * kotlin.math.sqrt(cx * cx + cy * cy + cz * cz)).coerceAtLeast(1e-9f)
            }
            val triOffsets = IntArray(FaceMeshTopology.FACE_VERTEX_COUNT + 1)
            for (i in tris.indices) triOffsets[tris[i].toInt() + 1]++
            for (v in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) triOffsets[v + 1] += triOffsets[v]
            val fill = triOffsets.copyOf()
            val triFlat = ShortArray(tris.size)
            for (i in tris.indices) triFlat[fill[tris[i].toInt()]++] = (i / 3).toShort()
            VERTEX_TRI_OFFSETS = triOffsets
            VERTEX_TRI_INDICES = triFlat
        }

        private fun smoothstep(e0: Float, e1: Float, x: Float): Float {
            val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
            return t * t * (3f - 2f * t)
        }

        private fun gaussian(x: Float, center: Float, sigma: Float): Float {
            val z = (x - center) / sigma
            return exp(-0.5f * z * z)
        }

        private fun radialFalloff(px: Float, py: Float, cx: Float, cy: Float, radius: Float): Float {
            if (radius <= 1e-4f) return 0f
            val r = hypot(px - cx, py - cy) / radius
            if (r >= 1f) return 0f
            val t = 1f - r * r
            return t * t
        }
    }
}
