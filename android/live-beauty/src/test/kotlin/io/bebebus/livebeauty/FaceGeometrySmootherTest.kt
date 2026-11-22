package io.bebebus.livebeauty

import io.bebebus.livebeauty.BeautyVectors.floats
import io.bebebus.livebeauty.BeautyVectors.list
import io.bebebus.livebeauty.BeautyVectors.number
import io.bebebus.livebeauty.BeautyVectors.obj
import org.junit.Assert.*
import org.junit.Test

/** 逐步回放共享向量 smoother 段（reference/beauty_ref.py 的 FinalSmoother），float32 容差 1e-4。 */
class FaceGeometrySmootherTest {
    private val section = BeautyVectors.section("smoother")
    private val tolerance = BeautyVectors.section("meta")["tolerance"].obj()["smoother"].number().toFloat()

    @Test fun stateLayoutMatchesReference() {
        assertEquals(FaceGeometrySmoother.STATE_SIZE, section["stateKinds"].list().size)
        assertEquals(FaceGeometrySmoother.RX_INDEX, section["rxIndex"].number().toInt())
        assertEquals(FaceGeometrySmoother.AXIS_INDEX, section["axisIndex"].number().toInt())
        assertEquals(FaceGeometrySmoother.WARP_INDEX, section["warpWeightIndex"].number().toInt())
    }

    @Test fun replaysSharedSequences() {
        val aspect = section["aspect"].number().toFloat()
        val names = mutableListOf<String>()
        for (item in section["sequences"].list()) {
            val sequence = item.obj(); val name = sequence["name"] as String
            names += name
            val samples = sequence["samples"].list().map { it.obj() }
            val smoother = FaceGeometrySmoother(aspect)
            for ((index, stepItem) in sequence["steps"].list().withIndex()) {
                val step = stepItem.obj(); val label = "$name#$index"
                if (step["op"] == "reset") { smoother.reset(); continue }
                val sample = step["sample"]?.let { samples[it.number().toInt()] }
                val actual = smoother.update(sample?.get("target")?.floats(),
                    sample?.get("sampleTs")?.number()?.toLong() ?: 0L, step["now"].number().toLong())
                val expected = step["expectedState"]
                if (expected == null) assertNull(label, actual)
                else assertArrayEquals(label, expected.floats(), actual, tolerance)
                assertEquals("$label.presence", step["expectedPresence"].number().toFloat(), smoother.presence, tolerance)
            }
        }
        assertEquals(listOf("static_jitter", "fast_move_then_stop", "dropouts_and_dt_edges", "reset",
            "head_turn_warp_weight", "warp_weight_only"), names)
    }

    @Test fun repeatedCallsWithinSameMillisecondReturnSameOutput() {
        val smoother = FaceGeometrySmoother(0.5625f)
        val target = FloatArray(FaceGeometrySmoother.STATE_SIZE) { 0.5f }.apply { this[26] = 1f; this[27] = 0f }
        smoother.update(target, 0, 0)
        val moved = target.copyOf().apply { this[0] = 0.6f }
        val first = smoother.update(moved, 100, 140)!!.copyOf()
        val presence = smoother.presence
        assertArrayEquals(first, smoother.update(moved, 100, 140), 0f)
        assertEquals(presence, smoother.presence, 0f)
    }
}
