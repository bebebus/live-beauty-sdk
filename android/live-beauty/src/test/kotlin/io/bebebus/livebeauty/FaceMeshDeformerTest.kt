package io.bebebus.livebeauty

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot

class FaceMeshDeformerTest {
    /** 基于 MediaPipe 规范 UV 构造居中正脸（可选 478 点含虹膜），aspect = 9/16。 */
    private fun canonicalMediaPipePoints(
        aspect: Float = 9f / 16f,
        centerX: Float = 0.5f,
        centerY: Float = 0.5f,
        scaleX: Float = 0.42f,
        scaleY: Float = 0.34f,
        yawShift: Float = 0f,
        withIris: Boolean = true,
    ): List<Pair<Float, Float>> {
        val uv = FaceMeshTopology.CANONICAL_UV
        val pts = ArrayList<Pair<Float, Float>>(if (withIris) 478 else 468)
        for (i in 0 until 468) {
            var u = uv[i * 2]
            val v = uv[i * 2 + 1]
            // 模拟真实 3D 偏航：鼻尖等前凸区域 (z > 0) 随转头横向移动，而双颊 (z ≈ 0) 按 cos 透视收窄，避免把远侧眼睛压成 0 宽。
            val dx = u - 0.5f
            val dy = v - 0.5f
            val noseRidge = (1f - hypot(dx / 0.18f, (dy - 0.05f) / 0.22f)).coerceIn(0f, 1f)
            val faceDome = (1f - hypot(dx / 0.52f, dy / 0.52f)).coerceIn(0f, 1f)
            val zNorm = 0.28f * faceDome + 0.34f * noseRidge
            val angle = yawShift * 0.72f
            val cosA = kotlin.math.cos(angle.toDouble()).toFloat()
            val sinA = kotlin.math.sin(angle.toDouble()).toFloat()
            u = 0.5f + dx * cosA + zNorm * sinA
            val x = centerX + (u - 0.5f) * scaleX
            val y = centerY + (v - 0.5f) * scaleY
            pts.add(x to y)
        }
        if (withIris) {
            fun addIris(outer: Int, inner: Int, top: Int, bot: Int) {
                val cx = 0.25f * (pts[outer].first + pts[inner].first + pts[top].first + pts[bot].first)
                val cy = 0.25f * (pts[outer].second + pts[inner].second + pts[top].second + pts[bot].second)
                val r = abs(pts[inner].first - pts[outer].first) * 0.22f
                pts.add(cx to cy)
                pts.add((cx + r) to cy)
                pts.add(cx to (cy - r * aspect))
                pts.add((cx - r) to cy)
                pts.add(cx to (cy + r * aspect))
            }
            addIris(33, 133, 159, 145)
            addIris(362, 263, 386, 374)
        }
        return pts
    }

    @Test
    fun topologyIsWatertightWithZeroToothHoleAndConsistentWinding() {
        assertEquals(548, FaceMeshTopology.TOTAL_VERTEX_COUNT)
        assertEquals(1086 * 3, FaceMeshTopology.FULL_INDICES.size)
        assertEquals(14 * 3, FaceMeshTopology.RIGHT_EYE_INDICES.size)
        assertEquals(14 * 3, FaceMeshTopology.LEFT_EYE_INDICES.size)
        assertEquals(120 * 3, FaceMeshTopology.LIP_BAND_INDICES.size)
        // 验证唇带子网绝不含完全落在内唇孔洞内的三角形。
        val innerLip = setOf(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)
        for (i in FaceMeshTopology.LIP_BAND_INDICES.indices step 3) {
            val a = FaceMeshTopology.LIP_BAND_INDICES[i].toInt()
            val b = FaceMeshTopology.LIP_BAND_INDICES[i + 1].toInt()
            val c = FaceMeshTopology.LIP_BAND_INDICES[i + 2].toInt()
            assertFalse("内唇口腔不应出现唇带三角形 ($a,$b,$c)", a in innerLip && b in innerLip && c in innerLip)
        }
    }

    @Test
    fun faceMeshFrameSupports478PtsRelaxedSideProfileAndAsymmetricWeights() {
        val aspect = 9f / 16f
        val front = FaceMeshFrame.fromMediaPipe(canonicalMediaPipePoints(aspect = aspect), aspect)
        assertNotNull(front)
        assertTrue(front!!.hasIris)
        assertEquals(1f, front.leftWeight, 1e-4f)
        assertEquals(1f, front.rightWeight, 1e-4f)
        assertEquals(1f, front.yawWeight, 1e-4f)
        assertTrue(front.rightEyeOpen > 0.15f)
        assertTrue(front.leftEyeOpen > 0.15f)

        // 仅 468 点时仍可安全构建并自动合成虹膜占位。
        val noIris = FaceMeshFrame.fromMediaPipe(canonicalMediaPipePoints(aspect = aspect, withIris = false), aspect)
        assertNotNull(noIris)
        assertFalse(noIris!!.hasIris)

        // 中大侧脸（约 40° 转头）：旧 FaceGeometry.fromMesh 拒收（r > 0.45），但 FaceMeshFrame 接受并产出非对称权重。
        val sidePts = canonicalMediaPipePoints(aspect = aspect, yawShift = 1.0f)
        val side = FaceMeshFrame.fromMediaPipe(sidePts, aspect)
        assertNotNull("约 40° 侧脸应在 0.72 放宽范围内接受", side)
        assertNull("同一 40° 侧脸旧 FaceGeometry.fromMesh 应仍按 0.45 阈值拒收以保基线兼容", FaceGeometry.fromMesh(sidePts, aspect))
        assertTrue("转向画面右侧时 signedYaw > 0.45: ${side!!.signedYaw}", side.signedYaw > 0.45f)
        assertTrue("远侧（右侧）权重应明显低于近侧（左侧）", side.rightWeight < side.leftWeight - 0.5f)
        assertNotNull("侧脸应能产出兼容遮罩几何", side.toCompatGeometry(aspect))

        // 极端背脸（|r| > 0.72）拒绝。
        val extreme = FaceMeshFrame.fromMediaPipe(canonicalMediaPipePoints(aspect = aspect, yawShift = 1.65f), aspect)
        assertNull(extreme)
    }

    @Test
    fun faceTimelineSmoothsInterpolatesAndExtrapolates478Points() {
        val aspect = 9f / 16f
        val timeline = FaceTimeline().apply { this.aspect = aspect }
        val pts1 = canonicalMediaPipePoints(aspect = aspect, centerX = 0.45f)
        val pts2 = canonicalMediaPipePoints(aspect = aspect, centerX = 0.55f)
        val g1 = FaceGeometry.fromMesh(pts1, aspect)!!
        val m1 = FaceMeshFrame.fromMediaPipe(pts1, aspect)!!
        val g2 = FaceGeometry.fromMesh(pts2, aspect)!!
        val m2 = FaceMeshFrame.fromMediaPipe(pts2, aspect)!!

        timeline.add(100L, g1.toState(), m1)
        timeline.add(200L, g2.toState(), m2)
        timeline.update(200L)
        val midGeo = timeline.at(150L)
        val midMesh = timeline.meshAt(150L)
        assertNotNull(midGeo)
        assertNotNull(midMesh)
        assertTrue(midMesh!!.hasIris)
        // t=150 的鼻尖 x 应严格落在 t=100 与 t=200 之间。
        val nose1 = m1.landmarks[1 * 2]
        val noseMid = midMesh.landmarks[1 * 2]
        val nose2 = m2.landmarks[1 * 2]
        assertTrue("插值鼻尖应位于两样本之间：$nose1 < $noseMid < $nose2", noseMid > nose1 && noseMid < nose2)

        // 外推 t=240：鼻尖继续沿速度向右平移，且不抛错。
        timeline.update(240L)
        timeline.at(240L)
        val extMesh = timeline.meshAt(240L)!!
        assertTrue("外推应沿向右速度前进", extMesh.landmarks[1 * 2] > noseMid)
    }

    @Test
    fun faceTimelineTracksBlinkAndInPlaceYawWithoutTrailingGhost() {
        val aspect = 9f / 16f
        val timeline = FaceTimeline().apply { this.aspect = aspect }
        // 连续 4 帧静止正面，让时间线稳态。
        val basePts = canonicalMediaPipePoints(aspect = aspect)
        val baseGeo = FaceGeometry.fromMesh(basePts, aspect)!!
        val baseMesh = FaceMeshFrame.fromMediaPipe(basePts, aspect)!!
        for (k in 0..3) {
            timeline.add(k * 33L, baseGeo.toState(), baseMesh)
            timeline.update(k * 33L)
        }
        // 第 4 帧：脸心原地不动，但突然眨眼（上下眼睑闭拢）+ 突然原地转头（yawShift = 0.55）。
        val turnPts = canonicalMediaPipePoints(aspect = aspect, yawShift = 0.55f).toMutableList()
        // 整条上眼睑落到下眼睑上（各对上下睑点收到中点 ±0.0004）。
        val lidPairs = intArrayOf(161, 163, 160, 144, 159, 145, 158, 153, 157, 154, 388, 390, 387, 373, 386, 374, 385, 380, 384, 381)
        for (k in lidPairs.indices step 2) {
            val up = lidPairs[k]; val down = lidPairs[k + 1]
            val midX = 0.5f * (turnPts[up].first + turnPts[down].first)
            val midY = 0.5f * (turnPts[up].second + turnPts[down].second)
            turnPts[up] = midX to (midY - 0.0004f)
            turnPts[down] = midX to (midY + 0.0004f)
        }
        val turnMesh = FaceMeshFrame.fromMediaPipe(turnPts, aspect)!!
        val turnGeo = FaceGeometry.fromMesh(turnPts, aspect) ?: turnMesh.toCompatGeometry(aspect)!!
        assertTrue("原始眨眼开合应 < 0.06", turnMesh.rightEyeOpen < 0.06f && turnMesh.leftEyeOpen < 0.06f)

        timeline.add(4 * 33L, turnGeo.toState(), turnMesh)
        timeline.update(4 * 33L)
        timeline.at(4 * 33L)
        val tracked = timeline.meshAt(4 * 33L)!!
        // 1. 眨眼门控：闭眼的第一个样本两眼门控即为 0（美瞳立即收起不穿帮）。
        assertTrue("眨眼首个样本门控应为 0：R=${tracked.rightIrisGate}, L=${tracked.leftIrisGate}",
            tracked.rightIrisGate == 0f && tracked.leftIrisGate == 0f)
        // 2. 原地转头无 1 Hz 滞后拖尾：鼻尖首帧即追上至少 85% 位移。
        val noseDeltaTotal = abs(turnMesh.landmarks[1 * 2] - baseMesh.landmarks[1 * 2])
        val noseDeltaTracked = abs(tracked.landmarks[1 * 2] - baseMesh.landmarks[1 * 2])
        assertTrue("原地转头首帧应迅速跟上（目标 $noseDeltaTotal，实际 $noseDeltaTracked）",
            noseDeltaTracked >= noseDeltaTotal * 0.85f)
    }

    @Test
    fun all14ReshapeControlsDeformInCorrectDirectionWithZeroBackgroundBend() {
        val aspect = 9f / 16f
        val mesh = FaceMeshFrame.fromMediaPipe(canonicalMediaPipePoints(aspect = aspect), aspect)!!
        val deformer = FaceMeshDeformer()

        fun run(params: FaceMeshDeformer.Params): FloatArray {
            val ok = deformer.deform(
                landmarks = mesh.landmarks,
                aspect = aspect,
                axisX = mesh.axisX,
                axisY = mesh.axisY,
                faceWidth = mesh.faceWidth,
                faceHeight = mesh.faceHeight,
                leftWeight = mesh.leftWeight,
                rightWeight = mesh.rightWeight,
                yawWeight = mesh.yawWeight,
                presence = 1f,
                steadyFace = 1f,
                steadyEye = 1f,
                params = params,
            )
            assertTrue(ok)
            // 断言 1：缓冲环 (468..503) 与屏幕外框 (504..511) 位移严格恒为 0（背景墙面门框直线零弯曲）。
            for (v in FaceMeshTopology.BUFFER_OFFSET until FaceMeshTopology.TOTAL_VERTEX_COUNT) {
                val k = v * 4
                val dstX = (deformer.vertexData[k] + 1f) * 0.5f
                val dstY = (deformer.vertexData[k + 1] + 1f) * 0.5f
                val srcX = deformer.vertexData[k + 2]
                val srcY = deformer.vertexData[k + 3]
                assertEquals("缓冲/外框顶点 $v X 必须零位移", srcX, dstX, 1e-5f)
                assertEquals("缓冲/外框顶点 $v Y 必须零位移", srcY, dstY, 1e-5f)
            }
            // 断言 2：全部 1014 个三角形在变形后无翻面（有向面积严格 > 0）。
            assertNoFoldedTriangles(deformer.vertexData)
            return deformer.warpedLandmarks.copyOf()
        }

        val src = mesh.landmarks
        fun x(a: FloatArray, i: Int) = a[i * 2]
        fun y(a: FloatArray, i: Int) = a[i * 2 + 1]

        // 0. 中性参数逐点恒等。
        val neutral = run(FaceMeshDeformer.Params())
        for (i in src.indices) assertEquals(src[i], neutral[i], 1e-5f)

        // 1. 瘦脸 faceSlim：下颌点 172/397 间距变窄。
        val slim = run(FaceMeshDeformer.Params(faceSlim = 1f))
        assertTrue(x(slim, 397) - x(slim, 172) < x(src, 397) - x(src, 172) - 0.01f)

        // 2. V 脸 vFace：下颌角/下巴两侧 149/378 向内收尖。
        val vface = run(FaceMeshDeformer.Params(vFace = 1f))
        assertTrue(x(vface, 378) - x(vface, 149) < x(src, 378) - x(src, 149) - 0.008f)

        // 3. 窄脸 faceNarrow：全脸宽 454-234 变窄。
        val narrow = run(FaceMeshDeformer.Params(faceNarrow = 1f))
        assertTrue(x(narrow, 454) - x(narrow, 234) < x(src, 454) - x(src, 234) - 0.01f)

        // 4. 小脸 faceSmall：脸宽与脸高同时等比缩小。
        val small = run(FaceMeshDeformer.Params(faceSmall = 1f))
        assertTrue(x(small, 454) - x(small, 234) < x(src, 454) - x(src, 234) - 0.01f)
        assertTrue(y(small, 10) - y(small, 152) < y(src, 10) - y(src, 152) - 0.01f)

        // 5. 瘦颧骨 cheekboneSlim：颧弓 454-234 明显内收，下巴尖 152 不动。
        val cheek = run(FaceMeshDeformer.Params(cheekboneSlim = 1f))
        assertTrue(x(cheek, 454) - x(cheek, 234) < x(src, 454) - x(src, 234) - 0.01f)
        assertEquals(y(src, 152), y(cheek, 152), 1e-3f)

        // 6. 下巴 chinShape（双向）：+1 下巴尖下移拉长，-1 下巴尖上移缩短。
        val chinLong = run(FaceMeshDeformer.Params(chinShape = 1f))
        val chinShort = run(FaceMeshDeformer.Params(chinShape = -1f))
        assertTrue(y(chinLong, 152) < y(src, 152) - 0.005f)
        assertTrue(y(chinShort, 152) > y(src, 152) + 0.005f)

        // 7. 额头 foreheadShape（双向）：+1 发际线上移，-1 发际线下移。
        val fhHigh = run(FaceMeshDeformer.Params(foreheadShape = 1f))
        val fhLow = run(FaceMeshDeformer.Params(foreheadShape = -1f))
        assertTrue(y(fhHigh, 10) > y(src, 10) + 0.005f)
        assertTrue(y(fhLow, 10) < y(src, 10) - 0.005f)

        // 8. 大眼 eyeEnlarge：上下眼睑 159-145 明显撑开，且眉毛 105 保护不被大幅顶高。
        val bigEye = run(FaceMeshDeformer.Params(eyeEnlarge = 1f))
        val eyeGapBefore = y(src, 159) - y(src, 145)
        val eyeGapAfter = y(bigEye, 159) - y(bigEye, 145)
        assertTrue("大眼上下眼睑应明显撑开：$eyeGapBefore -> $eyeGapAfter", eyeGapAfter > eyeGapBefore * 1.15f)
        assertTrue("眉毛保护：眉峰 105 位移应远小于眼睑扩张", abs(y(bigEye, 105) - y(src, 105)) < (eyeGapAfter - eyeGapBefore))

        // 9. 眼距 eyeDistance（双向）：+1 双眼间距拉宽，-1 拉近（虹膜 468/473 同步跟随）。
        val wideEye = run(FaceMeshDeformer.Params(eyeDistance = 1f))
        val closeEye = run(FaceMeshDeformer.Params(eyeDistance = -1f))
        assertTrue(x(wideEye, 362) - x(wideEye, 133) > x(src, 362) - x(src, 133) + 0.003f)
        assertTrue(x(closeEye, 362) - x(closeEye, 133) < x(src, 362) - x(src, 133) - 0.003f)
        assertTrue("虹膜中心 473-468 应同步拉宽", x(wideEye, 473) - x(wideEye, 468) > x(src, 473) - x(src, 468) + 0.003f)

        // 10. 眼角 eyeCorner（双向）：+1 外眼角 33/263 上扬，-1 下垂。
        val catEye = run(FaceMeshDeformer.Params(eyeCorner = 1f))
        val puppyEye = run(FaceMeshDeformer.Params(eyeCorner = -1f))
        assertTrue(y(catEye, 33) > y(src, 33) + 0.002f && y(catEye, 263) > y(src, 263) + 0.002f)
        assertTrue(y(puppyEye, 33) < y(src, 33) - 0.002f && y(puppyEye, 263) < y(src, 263) - 0.002f)

        // 11. 瘦鼻 noseSlim：鼻翼 358-129 间距缩小。
        val slimNose = run(FaceMeshDeformer.Params(noseSlim = 1f))
        assertTrue(x(slimNose, 358) - x(slimNose, 129) < x(src, 358) - x(src, 129) - 0.004f)

        // 12. 长鼻 noseLength（双向）：+1 鼻尖下移拉长，-1 鼻尖上提缩短。
        val longNose = run(FaceMeshDeformer.Params(noseLength = 1f))
        val shortNose = run(FaceMeshDeformer.Params(noseLength = -1f))
        assertTrue(y(longNose, 1) < y(src, 1) - 0.002f)
        assertTrue(y(shortNose, 1) > y(src, 1) + 0.002f)

        // 13. 嘴型 mouthSize（双向）：+1 嘴宽 291-61 放大，-1 缩小。
        val bigMouth = run(FaceMeshDeformer.Params(mouthSize = 1f))
        val smallMouth = run(FaceMeshDeformer.Params(mouthSize = -1f))
        assertTrue(x(bigMouth, 291) - x(bigMouth, 61) > x(src, 291) - x(src, 61) + 0.004f)
        assertTrue(x(smallMouth, 291) - x(smallMouth, 61) < x(src, 291) - x(src, 61) - 0.004f)

        // 14. 微笑唇 mouthSmile：左右嘴角 61/291 上扬。
        val smile = run(FaceMeshDeformer.Params(mouthSmile = 1f))
        assertTrue(y(smile, 61) > y(src, 61) + 0.0015f && y(smile, 291) > y(src, 291) + 0.0015f)
    }

    @Test
    fun extremeCombinedReshapeNeverFoldsAnyTriangle() {
        val aspect = 9f / 16f
        val mesh = FaceMeshFrame.fromMediaPipe(canonicalMediaPipePoints(aspect = aspect), aspect)!!
        val deformer = FaceMeshDeformer()
        val combos = listOf(
            FaceMeshDeformer.Params(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f),
            FaceMeshDeformer.Params(1f, 1f, 1f, 1f, 1f, -1f, -1f, 1f, -1f, -1f, 1f, -1f, -1f, 1f),
            FaceMeshDeformer.Params(1f, 0f, 1f, 0f, 1f, -1f, 1f, 1f, -1f, 1f, 1f, 1f, -1f, 1f),
        )
        for (p in combos) {
            assertTrue(
                deformer.deform(
                    mesh.landmarks, aspect, mesh.axisX, mesh.axisY, mesh.faceWidth, mesh.faceHeight,
                    1f, 1f, 1f, 1f, 1f, 1f, p,
                ),
            )
            assertNoFoldedTriangles(deformer.vertexData)
        }
    }

    @Test
    fun neckLiftAndRetouchPackUseReal478Landmarks() {
        val aspect = 9f / 16f
        val mesh = FaceMeshFrame.fromMediaPipe(canonicalMediaPipePoints(aspect = aspect), aspect)!!
        val body = floatArrayOf(0.5f, 0.1f, 0f, -1f, 0.35f, 0.18f, 1f)
        val lifted = FloatArray(FaceMeshFrame.COORD_COUNT)
        NeckLift.applyLandmarks(mesh.landmarks, mesh.centerX, mesh.centerY, body, 0.3f, aspect, lifted)
        val expectedDy = 0.3f * 0.35f * 0.3f * aspect
        assertEquals(mesh.landmarks[1 * 2 + 1] + expectedDy, lifted[1 * 2 + 1], 1e-4f)
        assertEquals(mesh.landmarks[468 * 2 + 1] + expectedDy, lifted[468 * 2 + 1], 1e-4f)

        val retouchOut = FloatArray(FaceShapePass.VALUES)
        assertTrue(FaceShapePass.packRetouchFromMesh(lifted, mesh.axisX, mesh.axisY, aspect, 0.5f, 0.4f, 0.3f, 0.2f, retouchOut))
        // 几何变形槽（下巴/额头/瘦鼻）必须为 0，仅局部修饰槽非零。
        assertEquals(0f, retouchOut[3], 1e-6f)
        assertEquals(0f, retouchOut[7], 1e-6f)
        assertEquals(0f, retouchOut[11], 1e-6f)
        assertEquals(0.5f, retouchOut[28], 1e-6f)
        assertEquals(0.2f, retouchOut[31], 1e-6f)
        assertTrue(hypot(retouchOut[16] - lifted[33 * 2], retouchOut[17] - lifted[33 * 2 + 1]) < 0.08f)
    }

    @Test
    fun mirroredFrontCameraMeshDeformsInwardWithConsistentCwWinding() {
        val aspect = 9f / 16f
        val normalPts = canonicalMediaPipePoints(aspect = aspect)
        val normalMesh = FaceMeshFrame.fromMediaPipe(normalPts, aspect)!!
        assertFalse("标准规范坐标应为逆时针绕序 (CCW)", FaceMeshFrame.isFrontFaceCw(normalMesh.landmarks))

        val mirroredPts = normalPts.map { (1f - it.first) to it.second }
        val mirroredMesh = FaceMeshFrame.fromMediaPipe(mirroredPts, aspect)!!
        assertTrue("水平镜像前置镜头坐标应检测为顺时针绕序 (CW)", FaceMeshFrame.isFrontFaceCw(mirroredMesh.landmarks))

        val deformer = FaceMeshDeformer()
        fun runMirrored(params: FaceMeshDeformer.Params): FloatArray {
            assertTrue(
                deformer.deform(
                    mirroredMesh.landmarks, aspect, mirroredMesh.axisX, mirroredMesh.axisY,
                    mirroredMesh.faceWidth, mirroredMesh.faceHeight,
                    mirroredMesh.leftWeight, mirroredMesh.rightWeight, mirroredMesh.yawWeight,
                    1f, 1f, 1f, params,
                ),
            )
            assertTrue("镜像网格变形后应标记 frontFaceCw=true", deformer.frontFaceCw)
            // 全部 1014 个三角形（含缓冲环与屏幕外框 8 锚点）必须保持一致顺时针绕序 (area < 0)，零交叉穿刺。
            assertNoFoldedTriangles(deformer.vertexData, expectCw = true)
            return deformer.warpedLandmarks.copyOf()
        }

        val src = mirroredMesh.landmarks
        fun spanX(a: FloatArray, i: Int, j: Int) = abs(a[i * 2] - a[j * 2])
        fun y(a: FloatArray, i: Int) = a[i * 2 + 1]

        // 镜像坐标下所有瘦脸类滑杆必须向内收窄（严禁因法线叉积反号变成向外撑胖）。
        val slim = runMirrored(FaceMeshDeformer.Params(faceSlim = 1f))
        assertTrue("镜像瘦脸应内收", spanX(slim, 397, 172) < spanX(src, 397, 172) - 0.01f)

        val vface = runMirrored(FaceMeshDeformer.Params(vFace = 1f))
        assertTrue("镜像 V 脸应内收", spanX(vface, 378, 149) < spanX(src, 378, 149) - 0.008f)

        val narrow = runMirrored(FaceMeshDeformer.Params(faceNarrow = 1f))
        assertTrue("镜像窄脸应内收", spanX(narrow, 454, 234) < spanX(src, 454, 234) - 0.01f)

        val cheek = runMirrored(FaceMeshDeformer.Params(cheekboneSlim = 1f))
        assertTrue("镜像瘦颧骨应内收", spanX(cheek, 454, 234) < spanX(src, 454, 234) - 0.01f)

        val small = runMirrored(FaceMeshDeformer.Params(faceSmall = 1f))
        assertTrue("镜像小脸应内收", spanX(small, 454, 234) < spanX(src, 454, 234) - 0.01f)

        val wideEye = runMirrored(FaceMeshDeformer.Params(eyeDistance = 1f))
        assertTrue("镜像开眼距应拉宽双眼间距", spanX(wideEye, 362, 133) > spanX(src, 362, 133) + 0.003f)

        val smile = runMirrored(FaceMeshDeformer.Params(mouthSmile = 1f))
        assertTrue("镜像微笑唇应上扬双侧嘴角", y(smile, 61) > y(src, 61) + 0.0015f && y(smile, 291) > y(src, 291) + 0.0015f)

        // 镜像叠加满格滑杆仍零翻面。
        runMirrored(FaceMeshDeformer.Params(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f))
    }

    @Test
    fun perspective3dSelfOccludingYawWithMaxStackedSlidersNeverTearsOrGougesCheek() {
        val aspect = 9f / 16f
        val uv = FaceMeshTopology.CANONICAL_UV
        val deformer = FaceMeshDeformer()

        // 构造真实 3D 颅面球体 + 近距自拍透视投影：耳前/鬓角边缘 Z 向颅后弯曲（z < 0），转头 15°/28° 时远侧边缘三角形在 2D 投影中自然发生 3D 自遮挡背向折叠。
        for (deg in listOf(14.0, 26.0)) {
            val rad = Math.toRadians(deg)
            val cosA = kotlin.math.cos(rad).toFloat()
            val sinA = kotlin.math.sin(rad).toFloat()
            val pts = ArrayList<Pair<Float, Float>>(468)
            for (i in 0 until 468) {
                val dx = uv[i * 2] - 0.5f
                val dy = uv[i * 2 + 1] - 0.5f
                val r = hypot(dx / 0.48f, dy / 0.50f)
                // 半球颅面深度：中心鼻梁前凸，耳前鬓角外缘沿球面切线近乎垂直折向颅后，复现 MediaPipe 3D 真实边缘自遮挡折叠。
                val rn = (r / 1.01f).coerceAtMost(1f)
                val nose = (1f - hypot(dx / 0.16f, (dy - 0.04f) / 0.20f)).coerceIn(0f, 1f)
                val skullZ = 0.52f * kotlin.math.sqrt((1f - rn * rn).coerceAtLeast(0f)) + 0.14f * nose
                val rx = dx * cosA + skullZ * sinA
                val rz = -dx * sinA + skullZ * cosA
                val invDepth = 1.8f / (1.8f - rz)
                val x = 0.5f + rx * invDepth * 0.44f
                val y = 0.5f + dy * invDepth * 0.35f
                pts.add(x to y)
            }

            val mesh = FaceMeshFrame.fromMediaPipe(pts, aspect)
            assertNotNull("${deg}° 3D 透视侧脸应被正常接受", mesh)
            val lm = mesh!!.landmarks

            // 确认测试夹具确实复现了 3D 远侧边缘自遮挡背向三角形 (origArea <= 0)。
            val indices = FaceMeshTopology.FULL_INDICES
            var preFoldedCount = 0
            for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                val ia = indices[t * 3].toInt()
                val ib = indices[t * 3 + 1].toInt()
                val ic = indices[t * 3 + 2].toInt()
                val origArea = (lm[ib * 2] - lm[ia * 2]) * (lm[ic * 2 + 1] - lm[ia * 2 + 1]) -
                    (lm[ib * 2 + 1] - lm[ia * 2 + 1]) * (lm[ic * 2] - lm[ia * 2])
                if (origArea <= 0f) preFoldedCount++
            }
            assertTrue("${deg}° 3D 投影应包含远侧自遮挡背向三角形以覆盖真实缺陷场景", preFoldedCount > 0)

            val stacked = FaceMeshDeformer.Params(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f)
            assertTrue(
                deformer.deform(
                    lm, aspect, mesh.axisX, mesh.axisY, mesh.faceWidth, mesh.faceHeight,
                    mesh.leftWeight, mesh.rightWeight, mesh.yawWeight,
                    1f, 1f, 1f, stacked,
                ),
            )

            // 1. 所有原本正面可见 (origArea > 1e-6) 的主网格三角形变形后绝不能翻面（杜绝发丝黑楔撕裂半边脸）。
            val verts = deformer.vertexData
            for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                val ia = indices[t * 3].toInt()
                val ib = indices[t * 3 + 1].toInt()
                val ic = indices[t * 3 + 2].toInt()
                val origArea = (lm[ib * 2] - lm[ia * 2]) * (lm[ic * 2 + 1] - lm[ia * 2 + 1]) -
                    (lm[ib * 2 + 1] - lm[ia * 2 + 1]) * (lm[ic * 2] - lm[ia * 2])
                if (origArea > 1e-6f) {
                    val a = ia * 4
                    val b = ib * 4
                    val c = ic * 4
                    val newArea = (verts[b] - verts[a]) * (verts[c + 1] - verts[a + 1]) -
                        (verts[b + 1] - verts[a + 1]) * (verts[c] - verts[a])
                    assertTrue("${deg}° 正面可见三角形 #$t 变形后严禁翻面：newArea=$newArea", newArea > 0f)
                }
            }

            // 2. 任意顶点最大位移受 C∞ 软饱和保护约束，严禁跨越半张脸拉扯背景/黑发到鼻尖。
            val maxAllowedMetric = 0.15f * maxOf(mesh.faceWidth, mesh.faceHeight)
            val warped = deformer.warpedLandmarks
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val dMetric = hypot((warped[i * 2] - lm[i * 2]) * aspect, warped[i * 2 + 1] - lm[i * 2 + 1])
                assertTrue("${deg}° 顶点 #$i 位移超界：$dMetric > $maxAllowedMetric", dMetric <= maxAllowedMetric)
            }

            // 3. 相邻正面顶点位移差保持 C1 连续平滑，杜绝 iter>=8 单点归零造成的锯齿撕裂。
            val maxEdgeShear = 0.065f * maxOf(mesh.faceWidth, mesh.faceHeight)
            for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                val tri = intArrayOf(indices[t * 3].toInt(), indices[t * 3 + 1].toInt(), indices[t * 3 + 2].toInt())
                for (e in 0..2) {
                    val u = tri[e]
                    val v = tri[(e + 1) % 3]
                    val ddx = ((warped[u * 2] - lm[u * 2]) - (warped[v * 2] - lm[v * 2])) * aspect
                    val ddy = (warped[u * 2 + 1] - lm[u * 2 + 1]) - (warped[v * 2 + 1] - lm[v * 2 + 1])
                    val edgeShear = hypot(ddx, ddy)
                    assertTrue("${deg}° 边 ($u,$v) 位移跳变过大（撕裂风险）：$edgeShear > $maxEdgeShear", edgeShear <= maxEdgeShear)
                }
            }
        }
    }

    /** 用夹具网格的元数据跑一次形变（[nowMs] 缺省为无状态模式）。 */
    private fun deform(
        deformer: FaceMeshDeformer,
        landmarks: FloatArray,
        params: FaceMeshDeformer.Params,
        aspect: Float = FaceTestFixtures.PORTRAIT,
        nowMs: Long = Long.MIN_VALUE,
    ): Boolean {
        val meta = FloatArray(FaceMeshFrame.META_SIZE)
        assertTrue(FaceMeshFrame.fillMeta(landmarks, aspect, meta))
        return deformer.deform(
            landmarks, aspect, meta[FaceMeshFrame.META_AXIS_X], meta[FaceMeshFrame.META_AXIS_Y],
            meta[FaceMeshFrame.META_FACE_WIDTH], meta[FaceMeshFrame.META_FACE_HEIGHT],
            meta[FaceMeshFrame.META_LEFT_WEIGHT], meta[FaceMeshFrame.META_RIGHT_WEIGHT], meta[FaceMeshFrame.META_YAW_WEIGHT],
            1f, 1f, 1f, params, nowMs,
        )
    }

    /** 898 个规范三角形变形后（GPU 顶点缓冲）的有向面积（度量空间 px²@720）。 */
    private fun warpedAreas(vertices: FloatArray, aspect: Float): DoubleArray {
        val indices = FaceMeshTopology.FULL_INDICES
        return DoubleArray(FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) { t ->
            val a = indices[t * 3].toInt() * 4; val b = indices[t * 3 + 1].toInt() * 4; val c = indices[t * 3 + 2].toInt() * 4
            // NDC → 纹理坐标 × 0.5，y 再除以 aspect 到度量空间。
            val k = 0.5 * 0.5 * 720.0 * 720.0 / aspect
            0.5 * ((vertices[b] - vertices[a]) * (vertices[c + 1] - vertices[a + 1]) -
                (vertices[b + 1] - vertices[a + 1]) * (vertices[c] - vertices[a])).toDouble() * k
        }
    }

    @Test
    fun yawedPerspectiveMaxSlimNeverFoldsWithYawRobustVisibility() {
        // 真透视规范脸 15–40° 偏航（含水平镜像）+ 满档瘦脸 / 全部叠加，无状态模式：原本正面的三角形绝不翻面，
        // 位移受软饱和上限约束、相邻顶点不撕裂（v4.1「脸缺半块」两个根因不回归）。左右半脸权重一律取 1（不靠偏航渐出兜底，
        // 最严苛）；夹具约 33° 起偏航比超过 0.72，生产上 FaceMeshFrame 已拒收，这里直接喂形变器。
        // 14 项全部满档叠加只测到 35°：40° 时新旧实现都会剩一个亚像素细长三角形（规范面积约 1e-6）轻微翻面，属拒收范围外。
        val aspect = FaceTestFixtures.PORTRAIT
        val deformer = FaceMeshDeformer()
        val stacked = FaceMeshDeformer.Params(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f)
        val cases = listOf(15.0, 22.0, 30.0, 35.0, 40.0).flatMap { deg ->
            listOf(deg to FaceMeshDeformer.Params(faceSlim = 1f)) + if (deg <= 35.0) listOf(deg to stacked) else emptyList()
        }
        for ((deg, params) in cases) for (mirror in listOf(false, true)) {
            val lm = FaceTestFixtures.render(yawRad = Math.toRadians(deg))
            if (mirror) for (i in 0 until FaceMeshFrame.LANDMARK_COUNT) lm[i * 2] = 1f - lm[i * 2]
            val sign = if (mirror) -1f else 1f
            // 横轴 u 指向 454、v = (−u.y, u.x) 指向额头 10（同 FaceMeshFrame.fillMeta，但不做偏航拒收）。
            var ux = lm[454 * 2] - lm[234 * 2]; var uy = (lm[454 * 2 + 1] - lm[234 * 2 + 1]) / aspect
            val fw = hypot(ux, uy); ux /= fw; uy /= fw
            val fx = lm[10 * 2] - lm[152 * 2]; val fy = (lm[10 * 2 + 1] - lm[152 * 2 + 1]) / aspect
            if (-uy * fx + ux * fy < 0f) { ux = -ux; uy = -uy }
            val fh = abs(-uy * fx + ux * fy)
            assertTrue("$deg° mirror=$mirror", deformer.deform(lm, aspect, ux, uy, fw, fh, 1f, 1f, 1f, 1f, 1f, 1f, params))
            assertEquals(mirror, deformer.frontFaceCw)
            val verts = deformer.vertexData
            val indices = FaceMeshTopology.FULL_INDICES
            var backFacing = 0
            for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                val ia = indices[t * 3].toInt(); val ib = indices[t * 3 + 1].toInt(); val ic = indices[t * 3 + 2].toInt()
                val origArea = ((lm[ib * 2] - lm[ia * 2]) * (lm[ic * 2 + 1] - lm[ia * 2 + 1]) -
                    (lm[ib * 2 + 1] - lm[ia * 2 + 1]) * (lm[ic * 2] - lm[ia * 2])) * sign
                if (origArea <= 1e-6f) { backFacing++; continue }
                val a = ia * 4; val b = ib * 4; val c = ic * 4
                val newArea = ((verts[b] - verts[a]) * (verts[c + 1] - verts[a + 1]) -
                    (verts[b + 1] - verts[a + 1]) * (verts[c] - verts[a])) * sign
                assertTrue("$deg° mirror=$mirror 正面三角形 #$t 变形后翻面：$newArea", newArea > 0f)
            }
            if (deg >= 30.0) assertTrue("$deg° 夹具应含远侧自遮挡背面三角形", backFacing > 0)
            val scale = maxOf(fw, fh)
            val warped = deformer.warpedLandmarks
            for (i in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                val d = hypot(warped[i * 2] - lm[i * 2], (warped[i * 2 + 1] - lm[i * 2 + 1]) / aspect)
                assertTrue("$deg° 顶点 #$i 位移 $d 超界", d <= 0.15f * scale)
            }
            for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) for (e in 0..2) {
                val u = indices[t * 3 + e].toInt(); val v = indices[t * 3 + (e + 1) % 3].toInt()
                val shear = hypot((warped[u * 2] - lm[u * 2]) - (warped[v * 2] - lm[v * 2]),
                    ((warped[u * 2 + 1] - lm[u * 2 + 1]) - (warped[v * 2 + 1] - lm[v * 2 + 1])) / aspect)
                assertTrue("$deg° 边 ($u,$v) 位移跳变 $shear 过大", shear <= 0.065f * scale)
            }
            // 可见度只随转头渐出远侧（夹具正偏航时解剖远侧为 454，镜像不改变哪一侧被遮挡）：近侧满可见，远侧明显衰减。
            val near = 234
            val far = 454
            if (deg >= 30.0) assertTrue("$deg° 远侧 ${deformer.vertexVisibility[far]} 应低于近侧 ${deformer.vertexVisibility[near]}",
                deformer.vertexVisibility[far] < deformer.vertexVisibility[near] - 0.2f)
        }
    }

    /** 回放夹具序列：返回每帧逐帧剔除状态翻转的三角形数与翻转面积（px²@720），以及可见度造成的逐帧最大位移跳动（px@720）。 */
    private fun replay(yawDeg: Double, filtered: Boolean, frames: Int = 120): Triple<Double, Double, List<Double>> {
        val aspect = FaceTestFixtures.PORTRAIT
        val truth = FaceTestFixtures.render(yawRad = Math.toRadians(yawDeg))
        val rng = Random(5)
        val timeline = FaceTimeline().apply { this.aspect = aspect }
        val deformer = FaceMeshDeformer()
        val params = FaceMeshDeformer.Params(faceSlim = 0.4f, eyeEnlarge = 0.38f)
        // 瘦脸 0.4 时各顶点的满可见内收量（px@720，与仿真 vis_run 同口径）。
        val uv = FaceMeshTopology.CANONICAL_UV
        val slim = DoubleArray(FaceMeshFrame.FACE_COUNT) { i ->
            val nu = abs((uv[i * 2] - 0.5) * 2); val nv = (0.5 - uv[i * 2 + 1]) * 2
            if (nu <= 0.18) 0.0 else {
                val tt = ((nu - 0.18) / 0.70).coerceIn(0.0, 1.0)
                0.4 * 0.088 * 0.45 * 720 * tt * tt * (3 - 2 * tt) * exp(-0.5 * ((nv + 0.32) / 0.38).let { it * it })
            }
        }
        var flips = 0.0; var flipArea = 0.0; var counted = 0
        val jumps = ArrayList<Double>()
        var prevAreas: DoubleArray? = null
        var prevVis: FloatArray? = null
        for (k in 0 until frames + 45) {
            val ts = k * 33L
            val input = FaceTestFixtures.noisy(truth, 0.5, rng)
            val lm = if (filtered) {
                timeline.add(ts, FaceTestFixtures.state(input), FaceTestFixtures.frame(input))
                timeline.update(ts)
                timeline.meshAt(ts)!!.landmarks
            } else input
            assertTrue(deform(deformer, lm, params, nowMs = if (filtered) ts else Long.MIN_VALUE))
            if (k < 45) continue
            val areas = warpedAreas(deformer.vertexData, aspect)
            val vis = deformer.vertexVisibility.copyOf()
            prevAreas?.let { pa ->
                for (t in areas.indices) if ((areas[t] > 0) != (pa[t] > 0)) { flips++; flipArea += abs(areas[t]) }
                counted++
            }
            prevVis?.let { pv -> jumps += vis.indices.maxOf { abs(vis[it] - pv[it]) * slim[it] } }
            prevAreas = areas; prevVis = vis
        }
        return Triple(flips / counted, flipArea / counted, jumps)
    }

    @Test
    fun triangleFlipCountReplay() {
        for (yaw in listOf(0.0, 15.0, 26.0)) {
            val (flips, area, _) = replay(yaw, filtered = true)
            val (rawFlips, rawArea, _) = replay(yaw, filtered = false)
            // 仿真：旧链路每帧 44–79 个、600–1000 px²；新链路 9–13 个、约 30 px²。
            assertTrue("$yaw° 每帧剔除翻转 $flips 个 / $area px²（原始无状态 $rawFlips / $rawArea）",
                flips <= 15 && area <= 60 && flips <= rawFlips / 3)
        }
    }

    @Test
    fun visibilityContinuityUnderNoise() {
        for (yaw in listOf(0.0, 15.0, 30.0)) {
            val jumps = replay(yaw, filtered = true).third.sorted()
            val p95 = jumps[(jumps.size * 0.95).toInt().coerceAtMost(jumps.size - 1)]
            assertTrue("$yaw° 可见度 × 瘦脸位移的逐帧最大跳动 p95 $p95 ≤ 0.5 px@720", p95 <= 0.5)
        }
    }

    @Test
    fun relaxBarrierIsContinuous() {
        // 满档叠加：把单个顶点扰动 1e-5，输出顶点缓冲的最大变化 ≤ 1e-3（不再有 relax 阈值与提前退出造成的跳变）。
        val stacked = FaceMeshDeformer.Params(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f)
        for (yaw in listOf(0.0, 26.0)) {
            val lm = FaceTestFixtures.render(yawRad = Math.toRadians(yaw))
            val deformer = FaceMeshDeformer()
            assertTrue(deform(deformer, lm, stacked))
            val base = deformer.vertexData.copyOf()
            for (v in intArrayOf(1, 33, 152, 172, 234, 397, 454)) for (axis in 0..1) {
                val moved = lm.copyOf().also { it[v * 2 + axis] += 1e-5f }
                assertTrue(deform(deformer, moved, stacked))
                val worst = base.indices.maxOf { abs(deformer.vertexData[it] - base[it]) }
                assertTrue("$yaw° 顶点 $v 轴 $axis 扰动 1e-5 → 输出变化 $worst", worst <= 1e-3f)
            }
        }
    }

    @Test
    fun temporalStateResetsOnResetTemporal() {
        val params = FaceMeshDeformer.Params(faceSlim = 1f, eyeEnlarge = 0.5f)
        val front = FaceTestFixtures.render()
        val turned = FaceTestFixtures.render(yawRad = Math.toRadians(30.0))
        val stateless = FaceMeshDeformer().also { assertTrue(deform(it, turned, params)) }.vertexData.copyOf()
        val deformer = FaceMeshDeformer()
        for (k in 0 until 10) assertTrue(deform(deformer, front, params, nowMs = k * 33L))
        // 正脸 → 30° 突变：时间平滑使本帧与无状态结果不同（可见度按 40 ms 下降、松弛按 120 ms 恢复）。
        assertTrue(deform(deformer, turned, params, nowMs = 330))
        assertTrue(stateless.indices.any { abs(deformer.vertexData[it] - stateless[it]) > 1e-4f })
        // resetTemporal 之后（新一轮）立即等于无状态结果；距上次超过 200 ms 同样重置。
        deformer.resetTemporal()
        assertTrue(deform(deformer, turned, params, nowMs = 363))
        for (i in stateless.indices) assertEquals(stateless[i], deformer.vertexData[i], 1e-6f)
        for (k in 0 until 5) assertTrue(deform(deformer, front, params, nowMs = 400 + k * 33L))
        assertTrue(deform(deformer, turned, params, nowMs = 400 + 4 * 33L + FaceMeshDeformer.TEMPORAL_RESET_MS + 1))
        for (i in stateless.indices) assertEquals(stateless[i], deformer.vertexData[i], 1e-6f)
    }

    /**
     * 生产链路回放（同 VideoEffectsFilter）：FaceTimeline.add → at / meshAt，取滤波后网格的左右 / 偏航权重、presence、steady，
     * 新一轮时 resetTemporal，形变传 nowMs（时间平滑）；同一输入另跑一个无状态实例对照。σ = 0.5 采样像素噪声，每帧 33 ms，
     * [speedDegPerS] 为 0 时静止正脸，否则在 0→30° 之间三角波往返。前 30 帧预热不计。
     * 返回 (时间平滑实例, 无状态实例) 中出现原本正面（|面积| > 1e-6）三角形翻面的帧数。
     */
    private fun temporalFoldReplay(params: FaceMeshDeformer.Params, speedDegPerS: Double, seed: Long, frames: Int = 150): Pair<Int, Int> {
        val aspect = FaceTestFixtures.PORTRAIT
        val rng = Random(seed)
        val timeline = FaceTimeline().apply { this.aspect = aspect }
        val temporal = FaceMeshDeformer()
        val stateless = FaceMeshDeformer()
        val indices = FaceMeshTopology.FULL_INDICES
        fun folded(lm: FloatArray, verts: FloatArray): Boolean {
            for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                val ia = indices[t * 3].toInt(); val ib = indices[t * 3 + 1].toInt(); val ic = indices[t * 3 + 2].toInt()
                val origArea = (lm[ib * 2] - lm[ia * 2]) * (lm[ic * 2 + 1] - lm[ia * 2 + 1]) -
                    (lm[ib * 2 + 1] - lm[ia * 2 + 1]) * (lm[ic * 2] - lm[ia * 2])
                if (origArea <= 1e-6f) continue
                val a = ia * 4; val b = ib * 4; val c = ic * 4
                if ((verts[b] - verts[a]) * (verts[c + 1] - verts[a + 1]) - (verts[b + 1] - verts[a + 1]) * (verts[c] - verts[a]) <= 0f) return true
            }
            return false
        }
        var run = timeline.runId
        var foldedTemporal = 0
        var foldedStateless = 0
        for (k in 0 until frames + 30) {
            val ts = k * 33L
            val phase = (ts * speedDegPerS / 1000.0) % 60.0
            val deg = if (phase < 30.0) phase else 60.0 - phase
            val input = FaceTestFixtures.noisy(FaceTestFixtures.render(yawRad = Math.toRadians(deg)), 0.5, rng)
            timeline.add(ts, FaceTestFixtures.state(input), FaceTestFixtures.frame(input))
            timeline.update(ts)
            timeline.at(ts)
            val mesh = timeline.meshAt(ts) ?: continue
            if (timeline.runId != run) { run = timeline.runId; temporal.resetTemporal() }
            val presence = timeline.presence
            if (presence <= 0f) continue
            val lm = mesh.landmarks
            val okTemporal = temporal.deform(
                lm, aspect, mesh.axisX, mesh.axisY, mesh.faceWidth, mesh.faceHeight, mesh.leftWeight, mesh.rightWeight, mesh.yawWeight,
                presence, timeline.steadyFace, timeline.steadyEye, params, nowMs = ts,
            )
            val okStateless = stateless.deform(
                lm, aspect, mesh.axisX, mesh.axisY, mesh.faceWidth, mesh.faceHeight, mesh.leftWeight, mesh.rightWeight, mesh.yawWeight,
                presence, timeline.steadyFace, timeline.steadyEye, params,
            )
            if (k < 30) continue
            if (okTemporal && folded(lm, temporal.vertexData)) foldedTemporal++
            if (okStateless && folded(lm, stateless.vertexData)) foldedStateless++
        }
        return foldedTemporal to foldedStateless
    }

    @Test
    fun temporalReplayNeverFoldsFrontTriangles() {
        // 生产路径恒传 nowMs。松弛系数的时间平滑必须作为守卫的起点：放在守卫与最后一道安全之后逐顶点回升，相邻顶点不同步，
        // 嘴角、鼻翼、唇缘的细长正面三角形会翻面（中等多项组合静止正脸约 8% 的帧，同输入无状态 0 帧）。
        val mid = FaceMeshDeformer.Params(faceSlim = 0.5f, vFace = 0.4f, faceSmall = 0.3f, eyeEnlarge = 0.5f, noseSlim = 0.3f, mouthSmile = 0.3f)
        val slimEye = FaceMeshDeformer.Params(faceSlim = 1f, eyeEnlarge = 1f)
        for ((name, params) in listOf("多项组合" to mid, "瘦脸+大眼满档" to slimEye)) for (speed in listOf(0.0, 60.0)) {
            val (folded, stateless) = temporalFoldReplay(params, speed, seed = 1)
            assertEquals("$name ${speed}°/s：时间平滑模式原本正面三角形翻面的帧数（无状态 $stateless）", 0, folded)
        }
        // 14 项全部满档超出守卫能力，无状态也会少量翻面：时间平滑不得比同输入的无状态更差。
        val stacked = FaceMeshDeformer.Params(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f)
        for (speed in listOf(0.0, 60.0)) {
            val (folded, stateless) = temporalFoldReplay(stacked, speed, seed = 1)
            assertTrue("14 项满档 ${speed}°/s：时间平滑翻面 $folded 帧，应不多于无状态 $stateless 帧", folded <= stateless)
        }
    }

    @Test
    fun sideProfileYawKeepsCompactOuterBufferRingAndAnchorTrianglesUnfolded() {
        val aspect = FaceTestFixtures.PORTRAIT
        val params = FaceMeshDeformer.Params(faceSlim = 1f, vFace = 0.8f, cheekboneSlim = 0.8f)
        val indices = FaceMeshTopology.FULL_INDICES
        for (mirrored in listOf(false, true)) {
            for (deg in listOf(-38.0, -25.0, -15.0, 0.0, 15.0, 25.0, 38.0)) {
                val lm = FaceTestFixtures.render(yawRad = Math.toRadians(deg), aspect = aspect)
                if (mirrored) {
                    for (i in 0 until FaceMeshFrame.LANDMARK_COUNT) lm[i * 2] = 1f - lm[i * 2]
                }
                val accepted = if (abs(deg) <= 25.0) {
                    FaceMeshFrame.fromMediaPipe(FaceTestFixtures.toMediaPipe(lm), aspect).also { assertNotNull(it) }
                } else {
                    null
                }
                var ux = lm[454 * 2] - lm[234 * 2]
                var uy = (lm[454 * 2 + 1] - lm[234 * 2 + 1]) / aspect
                val fw = accepted?.faceWidth ?: hypot(ux, uy)
                if (accepted != null) {
                    ux = accepted.axisX
                    uy = accepted.axisY
                } else {
                    ux /= fw
                    uy /= fw
                    val fx = lm[10 * 2] - lm[152 * 2]
                    val fy = (lm[10 * 2 + 1] - lm[152 * 2 + 1]) / aspect
                    if (-uy * fx + ux * fy < 0f) { ux = -ux; uy = -uy }
                }
                val fh = accepted?.faceHeight ?: abs(-uy * (lm[10 * 2] - lm[152 * 2]) + ux * ((lm[10 * 2 + 1] - lm[152 * 2 + 1]) / aspect))
                val lw = accepted?.leftWeight ?: 1f
                val rw = accepted?.rightWeight ?: 1f
                val yw = accepted?.yawWeight ?: 1f
                val deformer = FaceMeshDeformer()
                assertTrue(
                    deformer.deform(
                        landmarks = accepted?.landmarks ?: lm,
                        aspect = aspect,
                        axisX = ux,
                        axisY = uy,
                        faceWidth = fw,
                        faceHeight = fh,
                        leftWeight = lw,
                        rightWeight = rw,
                        yawWeight = yw,
                        presence = 1f,
                        steadyFace = 1f,
                        steadyEye = 1f,
                        params = params,
                    ),
                )
                val srcLm = accepted?.landmarks ?: lm
                val verts = deformer.vertexData
                val maxSpan = maxOf(fw, fh)
                val sign = if (mirrored) -1f else 1f
                val vx = -uy
                val vy = ux
                // 1. 双环 C1 紧支集缓冲环：内环（MID_BUFFER_OFFSET）由同一 2D 连续微分同胚场随动包裹皮肉外缘，外环（BUFFER_OFFSET）
                //    严格恒等零位移截断背景拉伸，且直至 ±38° 侧脸外扩方向始终与规范外法向正对齐（绝不向脸内倒插）。
                for (k in 0 until FaceMeshTopology.OVAL_VERTEX_COUNT) {
                    val ovalIdx = FaceMeshTopology.FACE_OVAL[k]
                    val ox = srcLm[ovalIdx * 2]
                    val oy = srcLm[ovalIdx * 2 + 1]

                    val mIdx = (FaceMeshTopology.MID_BUFFER_OFFSET + k) * 4
                    val mx = verts[mIdx + 2]
                    val my = verts[mIdx + 3]
                    val mDispMetric = hypot(
                        ((verts[mIdx] + 1f) * 0.5f - mx).toDouble(),
                        (((verts[mIdx + 1] + 1f) * 0.5f - my) / aspect).toDouble(),
                    ).toFloat()
                    assertTrue(
                        "yaw=$deg mirrored=$mirrored 内过渡环位移超界：$mDispMetric",
                        mDispMetric <= 0.15f * maxSpan,
                    )

                    val bIdx = (FaceMeshTopology.BUFFER_OFFSET + k) * 4
                    val bx = (verts[bIdx] + 1f) * 0.5f
                    val by = (verts[bIdx + 1] + 1f) * 0.5f
                    assertEquals(verts[bIdx + 2], bx, 1e-5f)
                    assertEquals(verts[bIdx + 3], by, 1e-5f)
                    val dx = bx - ox
                    val dy = (by - oy) / aspect
                    val dist = hypot(dx.toDouble(), dy.toDouble()).toFloat()
                    assertTrue(
                        "yaw=$deg mirrored=$mirrored oval[$k] 外缓冲距异常 dist=$dist maxSpan=$maxSpan",
                        dist in (0.10f * maxSpan)..(0.28f * maxSpan),
                    )
                    val mDist = hypot((mx - ox).toDouble(), ((my - oy) / aspect).toDouble()).toFloat()
                    assertTrue("内过渡环必须严格位于 FACE_OVAL 与外缓冲环之间：$mDist vs $dist", mDist > 0.03f * maxSpan && mDist < dist)
                    val cNu = (FaceMeshTopology.CANONICAL_UV[ovalIdx * 2] - 0.5f) * 2f * sign
                    val cNv = (0.5f - FaceMeshTopology.CANONICAL_UV[ovalIdx * 2 + 1]) * 2f
                    val outDot = dx * (cNu * ux + cNv * vx) + dy * (cNu * uy + cNv * vy)
                    assertTrue("yaw=$deg mirrored=$mirrored oval[$k] 缓冲方向向内倒插 outDot=$outDot", outDot > 0f)
                }
                // 2. 在 3D 太阳穴/耳际轮廓尚未越过相机视平线的 ±15° 内，全部 44 个外框锚点三角形绕序严格为正；
                //    在 ±25°..±38° 侧脸（太阳穴 284-251 / 耳际 58-132 自然越过视平线，由 FaceMeshWarpPass 外圈关背面剔除水密承接）下，
                //    8 个屏幕外框角扇形（1006..1013）仍全部严格为正，且全部 44 个缓冲/外框顶点严格恒等零位移。
                for (v in FaceMeshTopology.BUFFER_OFFSET until FaceMeshTopology.TOTAL_VERTEX_COUNT) {
                    val vi = v * 4
                    assertEquals(verts[vi + 2], (verts[vi] + 1f) * 0.5f, 1e-5f)
                    assertEquals(verts[vi + 3], (verts[vi + 1] + 1f) * 0.5f, 1e-5f)
                }
                val triStart = if (abs(deg) <= 15.0) {
                    FaceMeshTopology.DEFORMABLE_TRIANGLE_COUNT
                } else {
                    FaceMeshTopology.TOTAL_TRIANGLE_COUNT - FaceMeshTopology.OUTER_VERTEX_COUNT
                }
                for (t in triStart until FaceMeshTopology.TOTAL_TRIANGLE_COUNT) {
                    val a = indices[t * 3].toInt() * 4
                    val b = indices[t * 3 + 1].toInt() * 4
                    val c = indices[t * 3 + 2].toInt() * 4
                    val area = (verts[b] - verts[a]) * (verts[c + 1] - verts[a + 1]) -
                        (verts[b + 1] - verts[a + 1]) * (verts[c] - verts[a])
                    assertTrue("yaw=$deg mirrored=$mirrored 锚点三角形 #$t 翻面：area=$area", area * sign > 0f)
                }
            }
        }
    }

    @Test
    fun mildSelfieYawDeformsBothCheeksAndEyesProportionallyWithoutAsymmetry() {
        val aspect = FaceTestFixtures.PORTRAIT
        val params = FaceMeshDeformer.Params(faceSlim = 0.75f, vFace = 0.55f, eyeEnlarge = 0.65f)
        for (mirrored in listOf(false, true)) {
            for (deg in listOf(-18.0, -10.0, 10.0, 18.0)) {
                val lm = FaceTestFixtures.render(yawRad = Math.toRadians(deg), aspect = aspect)
                if (mirrored) {
                    for (i in 0 until FaceMeshFrame.LANDMARK_COUNT) lm[i * 2] = 1f - lm[i * 2]
                }
                val mesh = FaceMeshFrame.fromMediaPipe(FaceTestFixtures.toMediaPipe(lm), aspect)
                assertNotNull("yaw=$deg mirrored=$mirrored 应正常生成网格帧", mesh)
                mesh!!
                // 日常手持自拍偏角（<=20°）下双侧权重必须保持满额 1.0，杜绝单侧过早衰减导致大小脸/大小眼。
                assertTrue("yaw=$deg leftWeight=${mesh.leftWeight} 过低", mesh.leftWeight >= 0.98f)
                assertTrue("yaw=$deg rightWeight=${mesh.rightWeight} 过低", mesh.rightWeight >= 0.98f)

                val deformer = FaceMeshDeformer()
                assertTrue(
                    deformer.deform(
                        landmarks = mesh.landmarks,
                        aspect = aspect,
                        axisX = mesh.axisX,
                        axisY = mesh.axisY,
                        faceWidth = mesh.faceWidth,
                        faceHeight = mesh.faceHeight,
                        leftWeight = mesh.leftWeight,
                        rightWeight = mesh.rightWeight,
                        yawWeight = mesh.yawWeight,
                        presence = 1f,
                        steadyFace = 1f,
                        steadyEye = 1f,
                        params = params,
                    ),
                )
                val src = mesh.landmarks
                val dst = deformer.warpedLandmarks
                val sign = if (mirrored) -1f else 1f
                val midX = 0.35f * src[168 * 2] + 0.35f * src[2 * 2] + 0.30f * src[152 * 2]
                val midY = (0.35f * src[168 * 2 + 1] + 0.35f * src[2 * 2 + 1] + 0.30f * src[152 * 2 + 1]) / aspect
                fun projU(arr: FloatArray, idx: Int): Float =
                    (arr[idx * 2] - midX) * mesh.axisX + (arr[idx * 2 + 1] / aspect - midY) * mesh.axisY

                // 采集所有在屏幕真实可见（未被硬件背面剔除）的下颊段顶点，度量左右两侧真实渲染外轮廓相对正中矢状面的收缩比例。
                val visibleJaw = BooleanArray(FaceMeshTopology.FACE_VERTEX_COUNT)
                val tris = FaceMeshTopology.CANONICAL_INDICES
                for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                    val a = tris[t * 3].toInt()
                    val b = tris[t * 3 + 1].toInt()
                    val c = tris[t * 3 + 2].toInt()
                    val area = ((src[b * 2] - src[a * 2]) * (src[c * 2 + 1] - src[a * 2 + 1]) -
                        (src[b * 2 + 1] - src[a * 2 + 1]) * (src[c * 2] - src[a * 2])) * sign
                    if (area > 1e-6f) {
                        visibleJaw[a] = true; visibleJaw[b] = true; visibleJaw[c] = true
                    }
                }
                var rightEdge0 = 0f; var rightEdge1 = 0f
                var leftEdge0 = 0f; var leftEdge1 = 0f
                for (v in 0 until FaceMeshTopology.FACE_VERTEX_COUNT) {
                    if (!visibleJaw[v]) continue
                    val nv = (0.5f - FaceMeshTopology.CANONICAL_UV[v * 2 + 1]) * 2f
                    if (nv !in -0.68f..-0.12f) continue
                    rightEdge0 = maxOf(rightEdge0, projU(src, v))
                    rightEdge1 = maxOf(rightEdge1, projU(dst, v))
                    leftEdge0 = maxOf(leftEdge0, -projU(src, v))
                    leftEdge1 = maxOf(leftEdge1, -projU(dst, v))
                }
                val rShrink = (rightEdge0 - rightEdge1) / rightEdge0
                val lShrink = (leftEdge0 - leftEdge1) / leftEdge0
                assertTrue("yaw=$deg mirrored=$mirrored 双颊可见外轮廓均应有效内收：r=$rShrink l=$lShrink", rShrink > 0.025f && lShrink > 0.025f)
                val cheekRatio = maxOf(rShrink, lShrink) / minOf(rShrink, lShrink)
                assertTrue(
                    "yaw=$deg mirrored=$mirrored 左右可见外轮廓收缩比例失衡（大小脸）：r=$rShrink l=$lShrink ratio=$cheekRatio",
                    cheekRatio <= 1.55f,
                )

                val rEye0 = abs(src[159 * 2 + 1] - src[145 * 2 + 1])
                val rEye1 = abs(dst[159 * 2 + 1] - dst[145 * 2 + 1])
                val lEye0 = abs(src[386 * 2 + 1] - src[374 * 2 + 1])
                val lEye1 = abs(dst[386 * 2 + 1] - dst[374 * 2 + 1])
                val rEyeGrow = (rEye1 - rEye0) / rEye0
                val lEyeGrow = (lEye1 - lEye0) / lEye0
                assertTrue("yaw=$deg mirrored=$mirrored 双眼均应放大：r=$rEyeGrow l=$lEyeGrow", rEyeGrow > 0.08f && lEyeGrow > 0.08f)
                val eyeRatio = maxOf(rEyeGrow, lEyeGrow) / minOf(rEyeGrow, lEyeGrow)
                assertTrue(
                    "yaw=$deg mirrored=$mirrored 左右大眼比例失衡（大小眼）：r=$rEyeGrow l=$lEyeGrow ratio=$eyeRatio",
                    eyeRatio <= 1.25f,
                )
            }
        }
    }

    @Test
    fun contourScanlineRasterizerHasZeroDoubleEdgeGhostAndBalancesFacialAsymmetry() {
        val aspect = FaceTestFixtures.PORTRAIT
        val params = FaceMeshDeformer.Params(faceSlim = 0.85f, vFace = 0.65f)
        val fullTris = FaceMeshTopology.FULL_INDICES

        fun barycentric(px: Float, py: Float, ax: Float, ay: Float, bx: Float, by: Float, cx: Float, cy: Float): FloatArray? {
            val denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if (abs(denom) < 1e-11f) return null
            val w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            val w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            val w3 = 1f - w1 - w2
            return if (w1 >= -1e-5f && w2 >= -1e-5f && w3 >= -1e-5f) floatArrayOf(w1, w2, w3) else null
        }

        // 1. 复现 FaceMeshWarpPass 完整三步光栅化（quad 兜底 → 外圈 188 三角关剔除 → 前景 898 三角开背面剔除）：
        //    验证 0°、±10°、±18°（含真实脸颊皮肉/跟脸外延 8px）下：(a) 原背向三角形翻为正面的重影总面积 ≈ 0；
        //    (b) 扫描线严格只有左入、右出 2 次前景跳变，零双轮廓残影、零反向采样裂缝；(c) 渲染出的左右外轮廓收缩比例均衡。
        for (deg in listOf(-18.0, -10.0, 0.0, 10.0, 18.0)) {
            val lm = FaceTestFixtures.render(yawRad = Math.toRadians(deg), aspect = aspect)
            val mesh = FaceMeshFrame.fromMediaPipe(FaceTestFixtures.toMediaPipe(lm), aspect)!!
            val deformer = FaceMeshDeformer()
            assertTrue(
                deformer.deform(
                    mesh.landmarks, aspect, mesh.axisX, mesh.axisY, mesh.faceWidth, mesh.faceHeight,
                    mesh.leftWeight, mesh.rightWeight, mesh.yawWeight, 1f, 1f, 1f, params,
                ),
            )
            val v = deformer.vertexData
            fun srcX(idx: Int) = v[idx * 4 + 2]
            fun srcY(idx: Int) = v[idx * 4 + 3]
            fun dstX(idx: Int) = (v[idx * 4] + 1f) * 0.5f
            fun dstY(idx: Int) = (v[idx * 4 + 1] + 1f) * 0.5f

            var unculledBackAreaPx2 = 0.0
            for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                val a = fullTris[t * 3].toInt()
                val b = fullTris[t * 3 + 1].toInt()
                val c = fullTris[t * 3 + 2].toInt()
                val origArea = (srcX(b) - srcX(a)) * (srcY(c) - srcY(a)) - (srcY(b) - srcY(a)) * (srcX(c) - srcX(a))
                val newArea = (dstX(b) - dstX(a)) * (dstY(c) - dstY(a)) - (dstY(b) - dstY(a)) * (dstX(c) - dstX(a))
                if (origArea <= 0f && newArea > 0f) {
                    unculledBackAreaPx2 += 0.5 * newArea * 720.0 * (720.0 / aspect)
                }
            }
            assertTrue("yaw=$deg 背向三角反翻未剔除重影面积过大：$unculledBackAreaPx2 px²", unculledBackAreaPx2 <= 3.0)

            val yScan = 0.5f * (srcY(10) + srcY(152)) - 0.16f * mesh.faceHeight * aspect
            var trueLeftPx = 1080
            var trueRightPx = 0
            for (xPx in 80..1000) {
                val px = xPx / 1080f
                for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                    val a = fullTris[t * 3].toInt(); val b = fullTris[t * 3 + 1].toInt(); val c = fullTris[t * 3 + 2].toInt()
                    val oa = (srcX(b) - srcX(a)) * (srcY(c) - srcY(a)) - (srcY(b) - srcY(a)) * (srcX(c) - srcX(a))
                    if (oa > 0f && barycentric(px, yScan, srcX(a), srcY(a), srcX(b), srcY(b), srcX(c), srcY(c)) != null) {
                        if (xPx < trueLeftPx) trueLeftPx = xPx
                        if (xPx > trueRightPx) trueRightPx = xPx
                        break
                    }
                }
            }
            val midPx = ((0.35f * srcX(168) + 0.35f * srcX(2) + 0.30f * srcX(152)) * 1080f).toInt()

            for (extraPx in listOf(0f, 8f)) {
                val sampledSx = FloatArray(921)
                val isSkin = BooleanArray(921)
                for (i in 0..920) {
                    val xPx = 80 + i
                    val px = xPx / 1080f
                    var sx = px
                    for (t in FaceMeshTopology.CANONICAL_TRIANGLE_COUNT until FaceMeshTopology.TOTAL_TRIANGLE_COUNT) {
                        val a = fullTris[t * 3].toInt(); val b = fullTris[t * 3 + 1].toInt(); val c = fullTris[t * 3 + 2].toInt()
                        val w = barycentric(px, yScan, dstX(a), dstY(a), dstX(b), dstY(b), dstX(c), dstY(c))
                        if (w != null) sx = w[0] * srcX(a) + w[1] * srcX(b) + w[2] * srcX(c)
                    }
                    for (t in 0 until FaceMeshTopology.CANONICAL_TRIANGLE_COUNT) {
                        val a = fullTris[t * 3].toInt(); val b = fullTris[t * 3 + 1].toInt(); val c = fullTris[t * 3 + 2].toInt()
                        val da = (dstX(b) - dstX(a)) * (dstY(c) - dstY(a)) - (dstY(b) - dstY(a)) * (dstX(c) - dstX(a))
                        if (da <= 0f) continue
                        val w = barycentric(px, yScan, dstX(a), dstY(a), dstX(b), dstY(b), dstX(c), dstY(c))
                        if (w != null) sx = w[0] * srcX(a) + w[1] * srcX(b) + w[2] * srcX(c)
                    }
                    sampledSx[i] = sx * 1080f
                    isSkin[i] = sampledSx[i] in (trueLeftPx - extraPx)..(trueRightPx + extraPx)
                }
                var transitions = 0
                var outLeft = 1080
                var outRight = 0
                for (i in 0..920) {
                    if (i > 0 && isSkin[i] != isSkin[i - 1]) transitions++
                    if (isSkin[i]) {
                        val xPx = 80 + i
                        if (xPx < outLeft) outLeft = xPx
                        if (xPx > outRight) outRight = xPx
                    }
                    if (i > 0) {
                        val dsx = sampledSx[i] - sampledSx[i - 1]
                        assertTrue("yaw=$deg extra=$extraPx x=${80 + i} 出现双轮廓采样跳变 dsx=$dsx", abs(dsx - 1f) <= 2.0f)
                    }
                }
                assertEquals("yaw=$deg extra=$extraPx 扫描线必须仅含左右两次单调轮廓边界（无残影重边）", 2, transitions)
                val slimL = (outLeft - (trueLeftPx - extraPx)) / ((midPx - trueLeftPx) + extraPx)
                val slimR = ((trueRightPx + extraPx) - outRight) / ((trueRightPx - midPx) + extraPx)
                val ratio = maxOf(slimL, slimR) / minOf(slimL, slimR).coerceAtLeast(1e-4f)
                assertTrue("yaw=$deg extra=$extraPx 渲染左右脸收缩比例失衡：L=$slimL R=$slimR ratio=$ratio", ratio <= 1.30f)
            }
        }

        // 2. 正脸天然咬肌不对称（左半脸宽 12%）：瘦脸后更饱满的左脸内收像素量应严格大于右脸，主动收敛天生大小脸。
        val asymLm = FaceTestFixtures.render(yawRad = 0.0, aspect = aspect)
        val mid0 = asymLm[2 * 2]
        for (i in 0 until FaceMeshFrame.FACE_COUNT) {
            val dx = asymLm[i * 2] - mid0
            if (dx < 0f) asymLm[i * 2] = mid0 + dx * 1.12f
        }
        val asymMesh = FaceMeshFrame.fromMediaPipe(FaceTestFixtures.toMediaPipe(asymLm), aspect)!!
        val asymDeformer = FaceMeshDeformer()
        assertTrue(
            asymDeformer.deform(
                asymMesh.landmarks, aspect, asymMesh.axisX, asymMesh.axisY, asymMesh.faceWidth, asymMesh.faceHeight,
                asymMesh.leftWeight, asymMesh.rightWeight, asymMesh.yawWeight, 1f, 1f, 1f, params,
            ),
        )
        val leftMove = asymDeformer.warpedLandmarks[172 * 2] - asymMesh.landmarks[172 * 2]
        val rightMove = asymMesh.landmarks[397 * 2] - asymDeformer.warpedLandmarks[397 * 2]
        assertTrue("正脸天然左宽右窄时左颊内收量 ($leftMove) 应严格大于右颊 ($rightMove)", leftMove > rightMove * 1.15f)
    }

    private fun assertNoFoldedTriangles(vertices: FloatArray, expectCw: Boolean = false) {
        val sign = if (expectCw) -1f else 1f
        val indices = FaceMeshTopology.FULL_INDICES
        for (t in 0 until FaceMeshTopology.TOTAL_TRIANGLE_COUNT) {
            val a = indices[t * 3].toInt() * 4
            val b = indices[t * 3 + 1].toInt() * 4
            val c = indices[t * 3 + 2].toInt() * 4
            val area = (vertices[b] - vertices[a]) * (vertices[c + 1] - vertices[a + 1]) -
                (vertices[b + 1] - vertices[a + 1]) * (vertices[c] - vertices[a])
            assertTrue("三角形 #$t 发生翻面或退化：area=$area (expectCw=$expectCw)", area * sign > 0f)
        }
    }
}
