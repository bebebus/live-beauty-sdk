package io.bebebus.livebeauty

import kotlin.math.abs
import kotlin.math.hypot

/**
 * MediaPipe 478 点全网格帧快照（仅 Android，纯逻辑不依赖 Android/GL，不保存图像或身份）。
 *
 * 坐标约定与 [FaceGeometry] 一致：
 * - [landmarks] 长度为 [COORD_COUNT]（478 × 2），存放 GL 纹理坐标（x 向右、y 向上，[0, 1] 附近），
 *   其中 `0..467` 为规范人脸网格点，`468..472` 为画面左侧（解剖学右眼）虹膜，`473..477` 为画面右侧（解剖学左眼）虹膜；
 * - [axisX]/[axisY] 为度量空间 `(x, y/aspect)` 内横过面部的单位向量 u（指向画面右侧 454），`v = (-u.y, u.x)` 指向额头 10；
 * - [signedYaw] 放宽到 `|r| <= 0.72`（约 45°–50° 侧脸），并拆出左右半脸非对称可见权重 [leftWeight]/[rightWeight]，
 *   大角度侧脸时只渐出远侧轮廓，近侧轮廓与五官保持自然稳定。
 */
class FaceMeshFrame(
    val landmarks: FloatArray,
    val centerX: Float,
    val centerY: Float,
    val axisX: Float,
    val axisY: Float,
    val faceWidth: Float,
    val faceHeight: Float,
    val signedYaw: Float,
    val leftWeight: Float,
    val rightWeight: Float,
    val yawWeight: Float,
    val rightEyeOpen: Float,
    val leftEyeOpen: Float,
    val mouthOpen: Float,
    val hasIris: Boolean,
) {
    /**
     * 将标量元数据写入 [out]（长度 [META_SIZE]）。[FaceTimeline] 正常用滤波后网格经 [fillMeta] 重算，本方法只作兜底；
     * 两个虹膜门控写 1（由 FaceTimeline 用 FaceMeshFilter 的眨眼门控覆盖）。
     */
    fun writeMeta(out: FloatArray) {
        out[META_CENTER_X] = centerX
        out[META_CENTER_Y] = centerY
        out[META_AXIS_X] = axisX
        out[META_AXIS_Y] = axisY
        out[META_FACE_WIDTH] = faceWidth
        out[META_FACE_HEIGHT] = faceHeight
        out[META_SIGNED_YAW] = signedYaw
        out[META_LEFT_WEIGHT] = leftWeight
        out[META_RIGHT_WEIGHT] = rightWeight
        out[META_YAW_WEIGHT] = yawWeight
        out[META_RIGHT_EYE_OPEN] = rightEyeOpen
        out[META_LEFT_EYE_OPEN] = leftEyeOpen
        out[META_MOUTH_OPEN] = mouthOpen
        out[META_RIGHT_IRIS_GATE] = 1f
        out[META_LEFT_IRIS_GATE] = 1f
    }

    /**
     * 侧脸（0.45 < |r| <= 0.72）时 [FaceGeometry.fromMesh] 返回 null；本方法产出兼容的 41 维 [FaceGeometry]
     * （旧塑形权重 warpWeight 置 0，仅保留精准的椭圆/眼/唇皮肤保护遮罩，供主合成与局部修饰继续稳定工作）。
     */
    fun toCompatGeometry(aspect: Float): FaceGeometry? {
        if (!aspect.isFinite() || aspect <= 0f) return null
        val state = FloatArray(FaceGeometrySmoother.STATE_SIZE)
        if (!fillGeometryState(landmarks, axisX, axisY, aspect, 0f, state)) return null
        return FaceGeometry(
            cheeks = state.copyOfRange(0, 16),
            eyes = state.copyOfRange(16, 22),
            oval = state.copyOfRange(22, 26),
            axis = state.copyOfRange(26, 28),
            mouth = state.copyOfRange(28, 32),
            eyeMasks = state.copyOfRange(32, 40),
            warpWeight = state[40],
        )
    }

    companion object {
        const val FACE_COUNT = 468
        const val LANDMARK_COUNT = 478
        const val COORD_COUNT = LANDMARK_COUNT * 2
        const val MAX_YAW_RATIO = 0.72

        const val META_CENTER_X = 0
        const val META_CENTER_Y = 1
        const val META_AXIS_X = 2
        const val META_AXIS_Y = 3
        const val META_FACE_WIDTH = 4
        const val META_FACE_HEIGHT = 5
        const val META_SIGNED_YAW = 6
        const val META_LEFT_WEIGHT = 7
        const val META_RIGHT_WEIGHT = 8
        const val META_YAW_WEIGHT = 9
        const val META_RIGHT_EYE_OPEN = 10
        const val META_LEFT_EYE_OPEN = 11
        const val META_MOUTH_OPEN = 12
        /** 解剖右眼 / 左眼的眨眼门控（0 = 闭眼收起美瞳，1 = 睁眼），见 FaceMeshFilter。 */
        const val META_RIGHT_IRIS_GATE = 13
        const val META_LEFT_IRIS_GATE = 14
        const val META_SIZE = 15

        private val REQUIRED_CORE = intArrayOf(
            1, 132, 361, 172, 397, 33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14,
        )
        private val REQUIRED_EXT = intArrayOf(234, 454, 10, 152, 0, 17, 129, 358)

        private fun smoothstep(e0: Double, e1: Double, x: Double): Double {
            val t = ((x - e0) / (e1 - e0)).coerceIn(0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        }

        /**
         * 从 MediaPipe FaceLandmarker 输出（图像归一化坐标，y 向下）提取完整 478 点纹理坐标与侧脸/开合元数据。
         * 若输入仅 468 点（如测试桩），自动由双眼上下睑中心合成 10 个虹膜占位点并置 [hasIris] = false。
         */
        fun fromMediaPipe(points: List<Pair<Float, Float>>, aspect: Float): FaceMeshFrame? {
            if (points.size < FACE_COUNT || !aspect.isFinite() || aspect <= 0f) return null
            fun valid(i: Int, min: Float, max: Float): Boolean {
                val (x, y) = points[i]
                return x.isFinite() && y.isFinite() && x in min..max && y in min..max
            }
            if (REQUIRED_CORE.any { !valid(it, -0.05f, 1.05f) }) return null
            if (REQUIRED_EXT.any { !valid(it, -0.30f, 1.30f) }) return null
            for (i in 0 until FACE_COUNT) {
                val (x, y) = points[i]
                if (!x.isFinite() || !y.isFinite() || x !in -0.5f..1.5f || y !in -0.5f..1.5f) return null
            }

            val landmarks = FloatArray(COORD_COUNT)
            for (i in 0 until FACE_COUNT) {
                landmarks[i * 2] = points[i].first
                landmarks[i * 2 + 1] = 1f - points[i].second
            }
            val meta = FloatArray(META_SIZE)
            if (!fillMeta(landmarks, aspect, meta)) return null

            // 虹膜 468..477（若存在且全部有限）；否则由眼中心与眼宽合成占位点。
            var hasIris = points.size >= LANDMARK_COUNT
            if (hasIris) {
                for (i in FACE_COUNT until LANDMARK_COUNT) {
                    val (x, y) = points[i]
                    if (!x.isFinite() || !y.isFinite() || x !in -0.25f..1.25f || y !in -0.25f..1.25f) {
                        hasIris = false
                        break
                    }
                    landmarks[i * 2] = x
                    landmarks[i * 2 + 1] = 1f - y
                }
            }
            if (!hasIris) {
                synthesizeIris(landmarks, 468, 33, 133, 159, 145, aspect)
                synthesizeIris(landmarks, 473, 362, 263, 386, 374, aspect)
            }

            return FaceMeshFrame(
                landmarks = landmarks,
                centerX = meta[META_CENTER_X],
                centerY = meta[META_CENTER_Y],
                axisX = meta[META_AXIS_X],
                axisY = meta[META_AXIS_Y],
                faceWidth = meta[META_FACE_WIDTH],
                faceHeight = meta[META_FACE_HEIGHT],
                signedYaw = meta[META_SIGNED_YAW],
                leftWeight = meta[META_LEFT_WEIGHT],
                rightWeight = meta[META_RIGHT_WEIGHT],
                yawWeight = meta[META_YAW_WEIGHT],
                rightEyeOpen = meta[META_RIGHT_EYE_OPEN],
                leftEyeOpen = meta[META_LEFT_EYE_OPEN],
                mouthOpen = meta[META_MOUTH_OPEN],
                hasIris = hasIris,
            )
        }

        /**
         * 从 468 点纹理坐标（[landmarks] 至少 468 × 2，y 向上）计算脸中心、横轴、脸宽 / 高、有符号偏航、左右半脸权重、
         * 睁眼比与张嘴比，写入 [out]（长度 [META_SIZE]，两个虹膜门控写 1）。[fromMediaPipe] 用它处理原始点，
         * [FaceTimeline] 用它从滤波后的网格重算（元数据与网格同源，侧脸权重不再被偏航噪声逐帧调制）。
         * 尺寸退化或 |偏航比| > [MAX_YAW_RATIO] 时返回 false（[out] 内容未定义）。
         */
        fun fillMeta(landmarks: FloatArray, aspect: Float, out: FloatArray): Boolean {
            if (landmarks.size < FACE_COUNT * 2 || out.size < META_SIZE || !aspect.isFinite() || aspect <= 0f) return false
            val ratio = aspect.toDouble()

            fun mx(i: Int) = landmarks[i * 2].toDouble()
            fun my(i: Int) = landmarks[i * 2 + 1].toDouble() / ratio
            fun dist(a: Int, b: Int) = hypot(mx(a) - mx(b), my(a) - my(b))

            val span = dist(132, 361)
            if (span !in 0.08..0.90) return false

            val acrossX = mx(454) - mx(234)
            val acrossY = my(454) - my(234)
            val faceWidth = hypot(acrossX, acrossY)
            if (faceWidth < 0.05) return false
            var ux = acrossX / faceWidth
            var uy = acrossY / faceWidth
            val f10x = mx(10) - mx(152)
            val f10y = my(10) - my(152)
            if (-uy * f10x + ux * f10y < 0.0) {
                ux = -ux; uy = -uy
            }
            if (!ux.isFinite() || !uy.isFinite() || abs(hypot(ux, uy) - 1.0) >= 1e-5) return false
            val vx = -uy
            val vy = ux
            val faceHeight = abs(f10x * vx + f10y * vy)
            if (faceHeight < 0.05) return false

            // 有符号偏航比：鼻尖 1 相对两颊中点沿 +u（画面右侧）偏移比例；正值表示头转向画面右侧（右侧为远侧）。
            val cheekMidX = 0.5 * (mx(132) + mx(361))
            val cheekMidY = 0.5 * (my(132) + my(361))
            val signedYaw = ((mx(1) - cheekMidX) * ux + (my(1) - cheekMidY) * uy) / span
            if (abs(signedYaw) > MAX_YAW_RATIO) return false

            // 非对称侧脸权重：在常规自拍/手持偏角（|signedYaw| <= 0.36，约 24° 以内）保持左右双颊与双眼满权重 1.0，杜绝单手握持微侧头时出现大小脸/大小眼；
            // 仅在 > 24°..42° 大侧脸远侧轮廓临近自遮挡时平滑渐出。
            val rightWeight = 1.0 - smoothstep(0.36, 0.66, signedYaw)
            val leftWeight = 1.0 - smoothstep(0.36, 0.66, -signedYaw)
            val yawWeight = 1.0 - smoothstep(0.45, MAX_YAW_RATIO, abs(signedYaw))

            // 眼睑开合比（竖直睑裂 / 眼宽）与嘴部张开度。
            val rightEyeW = dist(33, 133)
            val leftEyeW = dist(362, 263)
            if (rightEyeW < 0.012 || leftEyeW < 0.012 || rightEyeW > span * 0.55 || leftEyeW > span * 0.55) return false
            val rightEyeOpen = (dist(159, 145) / rightEyeW).coerceIn(0.0, 1.0)
            val leftEyeOpen = (dist(386, 374) / leftEyeW).coerceIn(0.0, 1.0)
            val mouthW = dist(61, 291).coerceAtLeast(1e-4)
            val mouthOpen = (dist(13, 14) / mouthW).coerceIn(0.0, 1.5)

            // 脸中心（与 FaceGeometry.oval 一致的度量中点）。
            val sideMidX = 0.5 * (mx(234) + mx(454)); val sideMidY = 0.5 * (my(234) + my(454))
            val lineMidX = 0.5 * (mx(10) + mx(152)); val lineMidY = 0.5 * (my(10) + my(152))
            val cu = 0.5 * (sideMidX * ux + sideMidY * uy + lineMidX * ux + lineMidY * uy)
            val cv = lineMidX * vx + lineMidY * vy

            out[META_CENTER_X] = (cu * ux + cv * vx).toFloat()
            out[META_CENTER_Y] = ((cu * uy + cv * vy) * ratio).toFloat()
            out[META_AXIS_X] = ux.toFloat()
            out[META_AXIS_Y] = uy.toFloat()
            out[META_FACE_WIDTH] = faceWidth.toFloat()
            out[META_FACE_HEIGHT] = faceHeight.toFloat()
            out[META_SIGNED_YAW] = signedYaw.toFloat()
            out[META_LEFT_WEIGHT] = leftWeight.toFloat()
            out[META_RIGHT_WEIGHT] = rightWeight.toFloat()
            out[META_YAW_WEIGHT] = yawWeight.toFloat()
            out[META_RIGHT_EYE_OPEN] = rightEyeOpen.toFloat()
            out[META_LEFT_EYE_OPEN] = leftEyeOpen.toFloat()
            out[META_MOUTH_OPEN] = mouthOpen.toFloat()
            out[META_RIGHT_IRIS_GATE] = 1f
            out[META_LEFT_IRIS_GATE] = 1f
            return true
        }

        /**
         * 判断 468 点纹理坐标在 GL 屏幕空间是否呈顺时针（CW）正面绕序。
         * 利用横轴向量 (454 - 234) 与中轴向量 (10 - 152) 的二维叉积符号：对任意平面旋转与画幅宽高比严格不变；
         * 常规点集叉积 > 0（CCW 正面，返回 false），水平镜像点集叉积 < 0（CW 正面，返回 true）。
         */
        fun isFrontFaceCw(landmarks: FloatArray): Boolean {
            if (landmarks.size < FACE_COUNT * 2) return false
            val ax = landmarks[454 * 2] - landmarks[234 * 2]
            val ay = landmarks[454 * 2 + 1] - landmarks[234 * 2 + 1]
            val bx = landmarks[10 * 2] - landmarks[152 * 2]
            val by = landmarks[10 * 2 + 1] - landmarks[152 * 2 + 1]
            return ax * by - ay * bx < 0f
        }

        private fun synthesizeIris(
            landmarks: FloatArray,
            baseIdx: Int,
            outer: Int,
            inner: Int,
            top: Int,
            bottom: Int,
            aspect: Float,
        ) {
            val cx = 0.25f * (landmarks[outer * 2] + landmarks[inner * 2] + landmarks[top * 2] + landmarks[bottom * 2])
            val cy = 0.25f * (landmarks[outer * 2 + 1] + landmarks[inner * 2 + 1] + landmarks[top * 2 + 1] + landmarks[bottom * 2 + 1])
            val r = hypot(landmarks[outer * 2] - landmarks[inner * 2], (landmarks[outer * 2 + 1] - landmarks[inner * 2 + 1]) / aspect) * 0.22f
            landmarks[baseIdx * 2] = cx; landmarks[baseIdx * 2 + 1] = cy
            landmarks[(baseIdx + 1) * 2] = cx + r; landmarks[(baseIdx + 1) * 2 + 1] = cy
            landmarks[(baseIdx + 2) * 2] = cx; landmarks[(baseIdx + 2) * 2 + 1] = cy + r * aspect
            landmarks[(baseIdx + 3) * 2] = cx - r; landmarks[(baseIdx + 3) * 2 + 1] = cy
            landmarks[(baseIdx + 4) * 2] = cx; landmarks[(baseIdx + 4) * 2 + 1] = cy - r * aspect
        }

        /**
         * 从任意（原始或变形后的）478 点纹理坐标重算 41 维 [FaceGeometry] 状态向量，使主合成的皮肤保护遮罩
         * 与网格微整后的真实五官严格重合。
         */
        fun fillGeometryState(
            landmarks: FloatArray,
            axisX: Float,
            axisY: Float,
            aspect: Float,
            warpWeight: Float,
            out: FloatArray,
        ): Boolean {
            if (landmarks.size < FACE_COUNT * 2 || out.size < FaceGeometrySmoother.STATE_SIZE || !(aspect > 0f)) return false
            val n = hypot(axisX, axisY)
            if (n < 1e-4f) return false
            val ux = axisX / n; val uy = axisY / n
            val vx = -uy; val vy = ux

            fun lx(i: Int) = landmarks[i * 2]
            fun ly(i: Int) = landmarks[i * 2 + 1]
            fun mx(i: Int) = lx(i)
            fun my(i: Int) = ly(i) / aspect

            // cheeks[0..15]
            val pairs = intArrayOf(132, 361, 172, 397)
            for (p in 0 until 2) {
                val a = pairs[p * 2]; val b = pairs[p * 2 + 1]
                val midX = 0.5f * (lx(a) + lx(b)); val midY = 0.5f * (ly(a) + ly(b))
                val k0 = p * 8; val k1 = k0 + 4
                out[k0] = lx(a); out[k0 + 1] = ly(a); out[k0 + 2] = midX; out[k0 + 3] = midY
                out[k1] = lx(b); out[k1 + 1] = ly(b); out[k1 + 2] = midX; out[k1 + 3] = midY
            }
            // eyes[16..21]
            val eyePairs = intArrayOf(33, 133, 362, 263)
            for (i in 0 until 2) {
                val a = eyePairs[i * 2]; val b = eyePairs[i * 2 + 1]
                val ew = hypot(mx(a) - mx(b), my(a) - my(b))
                if (ew < 0.005f) return false
                out[16 + i * 3] = 0.5f * (lx(a) + lx(b))
                out[17 + i * 3] = 0.5f * (ly(a) + ly(b))
                out[18 + i * 3] = ew * 0.9f
            }
            fun centre(a: Int, b: Int, c: Int, d: Int, dest: Int) {
                val sx = 0.5f * (mx(a) + mx(b)); val sy = 0.5f * (my(a) + my(b))
                val tx = 0.5f * (mx(c) + mx(d)); val ty = 0.5f * (my(c) + my(d))
                val cu = 0.5f * (sx * ux + sy * uy + tx * ux + ty * uy)
                val cv = tx * vx + ty * vy
                out[dest] = cu * ux + cv * vx
                out[dest + 1] = (cu * uy + cv * vy) * aspect
            }
            val fw = hypot(mx(454) - mx(234), my(454) - my(234))
            val fh = abs((mx(10) - mx(152)) * vx + (my(10) - my(152)) * vy)
            if (fw < 0.03f || fh < 0.03f) return false
            centre(234, 454, 10, 152, 22)
            out[24] = 0.5f * fw * 1.12f
            out[25] = 0.5f * fh * 1.15f
            out[26] = ux; out[27] = uy

            centre(61, 291, 0, 17, 28)
            val mouthRx = 0.5f * abs((mx(61) - mx(291)) * ux + (my(61) - my(291)) * uy) * 1.20f
            if (mouthRx < 0.004f) return false
            out[30] = mouthRx
            out[31] = maxOf(0.5f * abs((mx(0) - mx(17)) * vx + (my(0) - my(17)) * vy) * 1.25f, 0.25f * mouthRx)

            for (i in 0 until 2) {
                val a = eyePairs[i * 2]; val b = eyePairs[i * 2 + 1]
                val ew = hypot(mx(a) - mx(b), my(a) - my(b))
                // 中心沿额头方向上移并扩大竖半径，将眉毛一并纳入保护区，防止美白/磨皮发灰冲淡眉色。
                val cxm = 0.5f * (mx(a) + mx(b)) + 0.16f * ew * vx
                val cym = 0.5f * (my(a) + my(b)) + 0.16f * ew * vy
                out[32 + i * 4] = cxm
                out[33 + i * 4] = cym * aspect
                out[34 + i * 4] = 0.78f * ew
                out[35 + i * 4] = 0.52f * ew
            }
            out[40] = warpWeight
            return true
        }
    }
}
