package io.bebebus.livebeauty

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import kotlin.math.hypot
import kotlin.math.max

/**
 * 468 点解剖子网格局部精修渲染通道（仅 Android，GLES 2.0，v1.1 升级）。
 *
 * 相比旧版全屏椭圆/直线近似 [FaceShapePass]，具备四大升级：
 * 1. **去黑眼圈精准留卧蚕**：静态顶点权重在下睑缘与卧蚕隆起带（`110,24,23,22,26,112` / `339,254,253,252,256,341`）严格置 0，
 *    仅在下方泪沟凹陷带（`228..233, 117..121` / `448..453, 346..350`）取峰值，并向下采样干净苹果肌参考肤色做暖调遮瑕，绝不抹平卧蚕；
 * 2. **弧形鼻唇沟贴合法令纹**：沿 4 段真实解剖鼻唇沟折线赋予权重与垂直跨沟法向，微笑大笑时紧贴弯曲鼻唇沟提亮凹陷暗线；
 * 3. **18 个内唇孔洞三角形物理防白牙染唇**：白牙严格只在 [FaceMakeupUv.INNER_MOUTH_INDICES] 上光栅化，唇红像素覆盖率为物理 `0.000`；
 * 4. **零额外 1080p FBO 开销**：直接在 [FaceMeshWarpPass.outputFbo] 上开启硬件混合绘制约 160 个局部微三角形（< 4% 屏幕面积），
 *    从纹理单元 0 读取未变形输入源纹理 `sourceTexture`，无读写反馈环且立省整张 1080p 中间纹理显存与全屏带宽。
 */
internal class FaceRetouchMeshPass {
    private var skinProg = 0
    private var skinPos = -1
    private var skinSrc = -1
    private var skinAttr = -1
    private var skinSampler = -1
    private var skinStrength = -1

    private var eyeTeethProg = 0
    private var etPos = -1
    private var etSrc = -1
    private var etAttr = -1
    private var etSampler = -1
    private var etMode = -1

    // 468 顶点 × 8 floats = (dstNdcX, dstNdcY, srcU, srcV, attrX, attrY, attrZ, attrW)
    private val vertexArray = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT * STRIDE_FLOATS)
    private val vertexBuffer = ByteBuffer
        .allocateDirect(vertexArray.size * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val skinRetouchIndexBuf: ShortBuffer = directShortBuffer(SKIN_RETOUCH_INDICES)
    private val eyeOpeningIndexBuf: ShortBuffer = directShortBuffer(FaceMakeupUv.EYE_OPENING_INDICES)
    private val innerMouthIndexBuf: ShortBuffer = directShortBuffer(FaceMakeupUv.INNER_MOUTH_INDICES)

    fun prepare() {
        ensurePrograms()
    }

    /**
     * 在 [targetFbo]（已包含网格变形或原画底图）上原位混合绘制 468 点局部精修子网格。
     *
     * @param sourceTexture 变形前的输入源纹理（与 [targetFbo] 颜色附件严格不同，零反馈环）
     * @param meshVertices 若本帧执行过 [FaceMeshDeformer.deform]，传入其 `vertexData`（含精确 `dstNdc` 与 `srcTex`）；
     *                     否则传 null，由 [landmarks] 同时生成恒等 `dstNdc` 与 `srcTex`。
     */
    fun renderInPlace(
        targetFbo: Int,
        sourceTexture: Int,
        width: Int,
        height: Int,
        landmarks: FloatArray,
        meshVertices: FloatArray?,
        axisX: Float,
        axisY: Float,
        faceWidth: Float,
        faceHeight: Float,
        frontFaceCw: Boolean,
        darkCircles: Float,
        brightEyes: Float,
        teethWhiten: Float,
        nasolabial: Float,
    ): Boolean {
        if (targetFbo == 0 || sourceTexture == 0 || width <= 0 || height <= 0) return false
        val dc = darkCircles.coerceIn(0f, 1f)
        val be = brightEyes.coerceIn(0f, 1f)
        val tw = teethWhiten.coerceIn(0f, 1f)
        val nf = nasolabial.coerceIn(0f, 1f)
        if (dc <= 1e-4f && be <= 1e-4f && tw <= 1e-4f && nf <= 1e-4f) return false
        val axisNorm = hypot(axisX, axisY)
        if (axisNorm < 1e-4f || faceWidth < 0.02f || faceHeight < 0.02f) return false
        val ux = axisX / axisNorm; val uy = axisY / axisNorm
        val vx = -uy; val vy = ux
        val aspect = width.toFloat() / height.toFloat()
        val mirrorSign = if (frontFaceCw) -1f else 1f

        val fresh = ensurePrograms()
        packVertices(landmarks, meshVertices, ux, uy, vx, vy, faceWidth, faceHeight, aspect, mirrorSign, dc, nf)
        vertexBuffer.clear()
        vertexBuffer.put(vertexArray)

        glBindFramebuffer(GL_FRAMEBUFFER, targetFbo)
        glViewport(0, 0, width, height)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_SCISSOR_TEST)
        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE)
        glEnable(GL_CULL_FACE)
        glFrontFace(if (frontFaceCw) GL_CW else GL_CCW)
        glCullFace(GL_BACK)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, sourceTexture)

        val byteStride = STRIDE_FLOATS * 4
        if (dc > 1e-4f || nf > 1e-4f) {
            glUseProgram(skinProg)
            bindAttribs(skinPos, skinSrc, skinAttr, byteStride)
            glUniform1i(skinSampler, 0)
            glUniform4f(skinStrength, dc, nf, 0f, 0f)
            skinRetouchIndexBuf.position(0)
            glDrawElements(GL_TRIANGLES, SKIN_RETOUCH_INDICES.size, GL_UNSIGNED_SHORT, skinRetouchIndexBuf)
            unbindAttribs(skinPos, skinSrc, skinAttr)
        }

        if (be > 1e-4f || tw > 1e-4f) {
            // 眼裂（28 三角形）与内唇口腔（18 三角形）子网格本身已是精确解剖边界，直接复用已上传的 vertexBuffer，省去二次 15 KB 拷贝。
            glUseProgram(eyeTeethProg)
            bindAttribs(etPos, etSrc, etAttr, byteStride)
            glUniform1i(etSampler, 0)
            if (be > 1e-4f) {
                glUniform4f(etMode, be, 0f, 0f, 0f)
                eyeOpeningIndexBuf.position(0)
                glDrawElements(GL_TRIANGLES, FaceMakeupUv.EYE_OPENING_INDICES.size, GL_UNSIGNED_SHORT, eyeOpeningIndexBuf)
            }
            if (tw > 1e-4f) {
                glUniform4f(etMode, tw, 1f, 0f, 0f)
                innerMouthIndexBuf.position(0)
                glDrawElements(GL_TRIANGLES, FaceMakeupUv.INNER_MOUTH_INDICES.size, GL_UNSIGNED_SHORT, innerMouthIndexBuf)
            }
            unbindAttribs(etPos, etSrc, etAttr)
        }

        glDisable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        glDisable(GL_BLEND)
        glBindTexture(GL_TEXTURE_2D, 0)
        if (fresh) check(glGetError() == GL_NO_ERROR)
        return true
    }

    internal val packedVerticesForTest: FloatArray get() = vertexArray

    internal fun packVertices(
        landmarks: FloatArray,
        meshVertices: FloatArray?,
        ux: Float,
        uy: Float,
        vx: Float,
        vy: Float,
        faceWidth: Float,
        faceHeight: Float,
        aspect: Float,
        mirrorSign: Float,
        dc: Float,
        nf: Float,
    ) {
        val needSkinOffsets = dc > 1e-4f || nf > 1e-4f
        // 泪沟参考偏移：沿面轴向下 (-v) 跨过泪沟采样苹果肌上沿干净肤色（度量步长 0.042 * fh）。
        val tearStepM = 0.042f * faceHeight
        val tearOffU = -vx * tearStepM
        val tearOffV = (-vy * tearStepM) * aspect
        val foldSpanM = 0.028f * faceWidth

        for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
            val k = i * STRIDE_FLOATS
            if (meshVertices != null) {
                val mk = i * 4
                vertexArray[k] = meshVertices[mk]
                vertexArray[k + 1] = meshVertices[mk + 1]
                vertexArray[k + 2] = meshVertices[mk + 2]
                vertexArray[k + 3] = meshVertices[mk + 3]
            } else {
                val sx = landmarks[i * 2]
                val sy = landmarks[i * 2 + 1]
                vertexArray[k] = sx * 2f - 1f
                vertexArray[k + 1] = sy * 2f - 1f
                vertexArray[k + 2] = sx
                vertexArray[k + 3] = sy
            }
            val wTear = STATIC_TEAR_WEIGHTS[i]
            val wFold = STATIC_FOLD_WEIGHTS[i]
            vertexArray[k + 4] = wTear
            vertexArray[k + 5] = wFold
            if (!needSkinOffsets || (wTear * dc <= 0f && wFold * nf <= 0f)) {
                vertexArray[k + 6] = 0f
                vertexArray[k + 7] = 0f
            } else if (wTear * dc >= wFold * nf) {
                vertexArray[k + 6] = tearOffU
                vertexArray[k + 7] = tearOffV
            } else {
                // 规范 UV (u 右, v 下) 法向转换至当前屏幕面轴度量方向 (ux, uy, vx, vy)。
                val cNu = STATIC_FOLD_NORMALS[i * 2] * mirrorSign
                val cNv = -STATIC_FOLD_NORMALS[i * 2 + 1]
                val dmX = (cNu * ux + cNv * vx) * foldSpanM
                val dmY = (cNu * uy + cNv * vy) * foldSpanM
                vertexArray[k + 6] = dmX
                vertexArray[k + 7] = dmY * aspect
            }
        }
    }

    private fun bindAttribs(posLoc: Int, srcLoc: Int, attrLoc: Int, byteStride: Int) {
        if (posLoc >= 0) {
            vertexBuffer.position(0)
            glVertexAttribPointer(posLoc, 2, GL_FLOAT, false, byteStride, vertexBuffer)
            glEnableVertexAttribArray(posLoc)
        }
        if (srcLoc >= 0) {
            vertexBuffer.position(2)
            glVertexAttribPointer(srcLoc, 2, GL_FLOAT, false, byteStride, vertexBuffer)
            glEnableVertexAttribArray(srcLoc)
        }
        if (attrLoc >= 0) {
            vertexBuffer.position(4)
            glVertexAttribPointer(attrLoc, 4, GL_FLOAT, false, byteStride, vertexBuffer)
            glEnableVertexAttribArray(attrLoc)
        }
    }

    private fun unbindAttribs(posLoc: Int, srcLoc: Int, attrLoc: Int) {
        if (posLoc >= 0) glDisableVertexAttribArray(posLoc)
        if (srcLoc >= 0) glDisableVertexAttribArray(srcLoc)
        if (attrLoc >= 0) glDisableVertexAttribArray(attrLoc)
    }

    private fun ensurePrograms(): Boolean {
        if (skinProg != 0 && eyeTeethProg != 0) return false
        if (skinProg == 0) {
            skinProg = GlUtil.createProgram(VideoEffectsShaders.RETOUCH_VERTEX, VideoEffectsShaders.RETOUCH_SKIN_FRAGMENT).also { check(it > 0) }
            skinPos = glGetAttribLocation(skinProg, "aPosition")
            skinSrc = glGetAttribLocation(skinProg, "aSrcCoord")
            skinAttr = glGetAttribLocation(skinProg, "aRetouchAttr")
            skinSampler = glGetUniformLocation(skinProg, "uSampler")
            skinStrength = glGetUniformLocation(skinProg, "uRetouchStrength")
        }
        if (eyeTeethProg == 0) {
            eyeTeethProg = GlUtil.createProgram(VideoEffectsShaders.RETOUCH_VERTEX, VideoEffectsShaders.RETOUCH_EYE_TEETH_FRAGMENT).also { check(it > 0) }
            etPos = glGetAttribLocation(eyeTeethProg, "aPosition")
            etSrc = glGetAttribLocation(eyeTeethProg, "aSrcCoord")
            etAttr = glGetAttribLocation(eyeTeethProg, "aRetouchAttr")
            etSampler = glGetUniformLocation(eyeTeethProg, "uSampler")
            etMode = glGetUniformLocation(eyeTeethProg, "uRetouchMode")
        }
        return true
    }

    fun release() {
        skinProg = 0
        eyeTeethProg = 0
    }

    fun delete() {
        if (skinProg != 0) glDeleteProgram(skinProg)
        if (eyeTeethProg != 0) glDeleteProgram(eyeTeethProg)
        release()
    }

    companion object {
        private const val STRIDE_FLOATS = 8

        /** 卧蚕保护点集（左右下睑紧邻的 12 个卧蚕隆起点）：去黑眼圈权重在此处强制锁 0，完整保留原生卧蚕立体感。 */
        internal val AEGYO_SAL_PROTECTED_SET = setOf(
            110, 24, 23, 22, 26, 112, 243,
            339, 254, 253, 252, 256, 341, 463,
        )

        private val LIP_PROTECTED_SET: Set<Int> = buildSet {
            addAll(FaceMakeupUv.INNER_LIP_SET)
            for (v in FaceMeshTopology.LIP_BAND_INDICES) add(v.toInt())
        }

        private val NOSE_TIP_PROTECTED_SET = setOf(1, 2, 4, 5, 19, 94, 168, 6, 197, 195)

        // 右侧与左侧泪沟弧线控制点（规范 UV，避开 v <= 0.413 的卧蚕隆起带，精准落在 v ≈ 0.425..0.462 泪沟凹陷槽）。
        private val RIGHT_TEAR_ARC = arrayOf(
            0.262f to 0.422f,
            0.298f to 0.440f,
            0.335f to 0.446f,
            0.372f to 0.436f,
            0.404f to 0.422f,
        )
        private val LEFT_TEAR_ARC = Array(RIGHT_TEAR_ARC.size) { i ->
            (1f - RIGHT_TEAR_ARC[i].first) to RIGHT_TEAR_ARC[i].second
        }

        // 右侧与左侧 4 段解剖鼻唇沟弧线（从鼻翼外缘 129/358 经 203/206/216 弯曲延伸至嘴角外侧 212/432）。
        private val RIGHT_FOLD_ARC = arrayOf(
            0.395f to 0.548f,
            0.378f to 0.574f,
            0.356f to 0.605f,
            0.337f to 0.646f,
            0.328f to 0.702f,
        )
        private val LEFT_FOLD_ARC = Array(RIGHT_FOLD_ARC.size) { i ->
            (1f - RIGHT_FOLD_ARC[i].first) to RIGHT_FOLD_ARC[i].second
        }

        internal val STATIC_TEAR_WEIGHTS = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
        internal val STATIC_FOLD_WEIGHTS = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT)
        internal val STATIC_FOLD_NORMALS = FloatArray(FaceMeshTopology.FACE_VERTEX_COUNT * 2)
        internal val SKIN_RETOUCH_INDICES: ShortArray

        init {
            val uv = FaceMeshTopology.CANONICAL_UV
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val u = uv[i * 2]
                val v = uv[i * 2 + 1]
                // 1. 泪沟权重（严格排除眼裂与卧蚕带）
                if (i !in FaceMakeupUv.RIGHT_EYE_SET && i !in FaceMakeupUv.LEFT_EYE_SET && i !in AEGYO_SAL_PROTECTED_SET) {
                    val dr = polylineDistAndNormal(u, v, RIGHT_TEAR_ARC).first
                    val dl = polylineDistAndNormal(u, v, LEFT_TEAR_ARC).first
                    val d = minOf(dr, dl)
                    val radius = 0.032f
                    if (d < radius && v >= 0.414f) {
                        val t = 1f - (d / radius) * (d / radius)
                        val lidGuard = smoothstep(0.414f, 0.426f, v)
                        STATIC_TEAR_WEIGHTS[i] = t * t * lidGuard
                    }
                }
                // 2. 弧形法令纹权重与局部跨沟法向（严格排除双唇与鼻头）
                if (i !in LIP_PROTECTED_SET && i !in NOSE_TIP_PROTECTED_SET) {
                    val (dr, nr) = polylineDistAndNormal(u, v, RIGHT_FOLD_ARC)
                    val (dl, nl) = polylineDistAndNormal(u, v, LEFT_FOLD_ARC)
                    val d = if (dr <= dl) dr else dl
                    val norm = if (dr <= dl) nr else nl
                    val radius = 0.028f
                    if (d < radius) {
                        val t = 1f - (d / radius) * (d / radius)
                        STATIC_FOLD_WEIGHTS[i] = t * t
                        STATIC_FOLD_NORMALS[i * 2] = norm.first
                        STATIC_FOLD_NORMALS[i * 2 + 1] = norm.second
                    }
                }
            }

            // 仅保留至少有一个顶点命中泪沟或法令纹权重的面部皮肤三角形（约 120 个微三角形，极省填充率）。
            val src = FaceMakeupUv.SKIN_FACE_INDICES
            val kept = ArrayList<Short>(420)
            for (t in src.indices step 3) {
                val a = src[t].toInt()
                val b = src[t + 1].toInt()
                val c = src[t + 2].toInt()
                if (max(max(STATIC_TEAR_WEIGHTS[a], STATIC_TEAR_WEIGHTS[b]), STATIC_TEAR_WEIGHTS[c]) > 0.002f ||
                    max(max(STATIC_FOLD_WEIGHTS[a], STATIC_FOLD_WEIGHTS[b]), STATIC_FOLD_WEIGHTS[c]) > 0.002f
                ) {
                    kept.add(src[t]); kept.add(src[t + 1]); kept.add(src[t + 2])
                }
            }
            SKIN_RETOUCH_INDICES = ShortArray(kept.size) { kept[it] }
        }

        private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
            val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
            return t * t * (3f - 2f * t)
        }

        private fun polylineDistAndNormal(u: Float, v: Float, arc: Array<Pair<Float, Float>>): Pair<Float, Pair<Float, Float>> {
            var bestDist = Float.MAX_VALUE
            var bestNx = 1f
            var bestNy = 0f
            for (s in 0 until arc.size - 1) {
                val (ax, ay) = arc[s]
                val (bx, by) = arc[s + 1]
                val abx = bx - ax
                val aby = by - ay
                val len2 = (abx * abx + aby * aby).coerceAtLeast(1e-8f)
                val t = (((u - ax) * abx + (v - ay) * aby) / len2).coerceIn(0f, 1f)
                val px = ax + abx * t
                val py = ay + aby * t
                val dist = hypot(u - px, v - py)
                if (dist < bestDist) {
                    bestDist = dist
                    val segLen = kotlin.math.sqrt(len2)
                    bestNx = -aby / segLen
                    bestNy = abx / segLen
                }
            }
            return bestDist to (bestNx to bestNy)
        }

        private fun directShortBuffer(src: ShortArray): ShortBuffer =
            ByteBuffer.allocateDirect(src.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer().apply {
                put(src)
                position(0)
            }
    }
}
