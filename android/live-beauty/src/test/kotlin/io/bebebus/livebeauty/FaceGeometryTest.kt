package io.bebebus.livebeauty

import io.bebebus.livebeauty.BeautyVectors.floats
import io.bebebus.livebeauty.BeautyVectors.list
import io.bebebus.livebeauty.BeautyVectors.number
import io.bebebus.livebeauty.BeautyVectors.obj
import org.junit.Assert.*
import org.junit.Test

class FaceGeometryTest {
    private fun mesh() = MutableList(478) { .5f to .5f }.apply {
        this[132]=.25f to .62f; this[361]=.75f to .62f
        this[172]=.33f to .76f; this[397]=.67f to .76f
        this[33]=.28f to .32f; this[133]=.42f to .32f
        this[362]=.58f to .32f; this[263]=.72f to .32f
        this[1]=.5f to .5f
        // 遮罩点：脸宽、额头/下巴、嘴角、上下唇。
        this[234]=.22f to .55f; this[454]=.78f to .55f
        this[10]=.5f to .15f; this[152]=.5f to .9f
        this[61]=.4f to .72f; this[291]=.6f to .72f
        this[0]=.5f to .69f; this[17]=.5f to .76f
    }
    @Test fun mapsUnmirroredAndMirroredMeshesToSameTextureCoordinates() {
        val geometry=FaceGeometry.fromMesh(mesh(),1f)!!
        assertEquals(.25f,geometry.cheeks[0],.00001f)
        assertEquals(.38f,geometry.cheeks[1],.00001f)
        assertEquals(.35f,geometry.eyes[0],.00001f)
        assertEquals(.68f,geometry.eyes[1],.00001f)
        val mirrored=FaceGeometry.fromMesh(mesh().map { 1-it.first to it.second },1f)!!
        assertEquals(.75f,mirrored.cheeks[0],.00001f)
        assertEquals(.65f,mirrored.eyes[0],.00001f)
        // 镜像后 p454 − p234 反向，axis 按 v 指向额头取符号，仍为 (1, 0)。
        assertArrayEquals(floatArrayOf(1f,0f),geometry.axis,.00001f)
        assertArrayEquals(floatArrayOf(1f,0f),mirrored.axis,.00001f)
        assertArrayEquals(geometry.oval,mirrored.oval.copyOf().also { it[0]=1-it[0] },.00001f)
    }
    @Test fun derivesOvalMouthAndEyeMasks() {
        val geometry=FaceGeometry.fromMesh(mesh(),1f)!!
        // 中心 u 分量取两对中点平均，v 分量取中线中点；半径为度量值。
        assertArrayEquals(floatArrayOf(.5f,.475f,.28f*1.12f,.375f*1.15f),geometry.oval,.00001f)
        assertArrayEquals(floatArrayOf(.5f,.275f,.1f*1.2f,.035f*1.25f),geometry.mouth,.00001f)
        assertArrayEquals(floatArrayOf(.35f,.68f+.04f*.14f,.72f*.14f,.36f*.14f,
            .65f,.68f+.04f*.14f,.72f*.14f,.36f*.14f),geometry.eyeMasks,.00001f)
        assertEquals(1f,geometry.warpWeight,0f)
        assertEquals(41,geometry.toState().size)
        assertEquals(geometry.warpWeight,geometry.toState()[40],0f)
        // 下唇距离很小时 ry 不小于 0.25 × rx。
        val flat=mesh().apply { this[0]=.5f to .72f; this[17]=.5f to .721f }
        val mouth=FaceGeometry.fromMesh(flat,1f)!!.mouth
        assertEquals(.25f*mouth[2],mouth[3],.00001f)
    }
    @Test fun rejectsIncompleteNonFiniteOutOfFrameAndExtremeYaw() {
        assertNull(FaceGeometry.fromMesh(mesh().take(100),1f))
        for (value in listOf(Float.NaN,Float.POSITIVE_INFINITY,-.1f,1.1f)) {
            val points=mesh();points[33]=value to .3f
            assertNull(FaceGeometry.fromMesh(points,1f))
        }
        // r = 2·|nose.x − 0.5| = 0.52 > 0.45。
        val turned=mesh();turned[1]=.76f to .5f
        assertNull(FaceGeometry.fromMesh(turned,1f))
        assertNull(FaceGeometry.fromMesh(mesh(),0f))
    }
    @Test fun maskPointsMayLeaveFrameWithinQuarterMargin() {
        for ((index,point) in listOf(10 to (.5f to -.25f),152 to (.5f to 1.25f),234 to (-.25f to .55f),454 to (1.25f to .55f))) {
            val points=mesh();points[index]=point
            assertNotNull("$index=$point",FaceGeometry.fromMesh(points,1f))
        }
        for ((index,point) in listOf(10 to (.5f to -.26f),152 to (.5f to 1.26f),234 to (-.26f to .55f),
                454 to (1.26f to .55f),61 to (Float.NaN to .72f),17 to (.5f to Float.POSITIVE_INFINITY))) {
            val points=mesh();points[index]=point
            assertNull("$index=$point",FaceGeometry.fromMesh(points,1f))
        }
        // 既有塑形点仍要求 [0, 1]。
        val cheek=mesh();cheek[132]=-.01f to .62f
        assertNull(FaceGeometry.fromMesh(cheek,1f))
    }
    @Test fun rejectsDegenerateMaskGeometry() {
        // 原 fixture 把 8 个遮罩点都放在 (0.5, 0.5)：脸宽、椭圆和嘴都退化为 0。
        val collapsed=mesh();for (i in listOf(234,454,10,152,61,291,0,17)) collapsed[i]=.5f to .5f
        assertNull(FaceGeometry.fromMesh(collapsed,1f))
        assertNull(FaceGeometry.fromMesh(mesh().apply { this[234]=.48f to .55f; this[454]=.52f to .55f },1f))
        assertNull(FaceGeometry.fromMesh(mesh().apply { this[10]=.5f to .5f; this[152]=.5f to .54f },1f))
        assertNull(FaceGeometry.fromMesh(mesh().apply { this[61]=.498f to .72f; this[291]=.502f to .72f },1f))
    }
    @Test fun sideFaceFadesWarpWeightBeforeRejecting() {
        fun turned(noseX: Float) = FaceGeometry.fromMesh(mesh().apply { this[1]=noseX to .5f },1f)
        assertEquals(1f,turned(.6f)!!.warpWeight,0f)        // r = 0.20
        assertEquals(.5f,turned(.635f)!!.warpWeight,.0001f) // r = 0.27，smoothstep 中点
        val faded=turned(.7f)!!                             // r = 0.40：几何有效，只停塑形
        assertEquals(0f,faded.warpWeight,0f)
        assertEquals(.5f,faded.oval[0],.00001f)
        assertNull(turned(.73f))                            // r = 0.46
    }
    @Test fun matchesSharedGeometryVectors() {
        val section=BeautyVectors.section("geometry")
        val tolerance=BeautyVectors.section("meta")["tolerance"].obj()["geometry"].number().toFloat()
        val cases=section["cases"].list()
        assertTrue(cases.size >= 20)
        for (item in cases) {
            val case=item.obj(); val name=case["name"] as String
            val points=MutableList(478) { .5f to .5f }
            for ((key,value) in case["landmarks"].obj()) {
                val xy=value.list()
                points[key.toInt()]=xy[0].number().toFloat() to xy[1].number().toFloat()
            }
            val actual=FaceGeometry.fromMesh(points,case["aspect"].number().toFloat())
            val expected=case["expected"]
            if (expected == null) { assertNull(name,actual); continue }
            assertNotNull(name,actual)
            val want=expected.obj()
            for ((field,value) in listOf("cheeks" to actual!!.cheeks,"eyes" to actual.eyes,"oval" to actual.oval,
                    "axis" to actual.axis,"mouth" to actual.mouth,"eyeMasks" to actual.eyeMasks))
                assertArrayEquals("$name.$field",want[field].floats(),value,tolerance)
            assertEquals("$name.warpWeight",want["warpWeight"].number().toFloat(),actual.warpWeight,tolerance)
        }
    }
    @Test fun singleFlightLimitAndSampleClockExpiry() {
        val state=FaceTrackingState();state.enable(true)
        val ticket=state.begin(1000,720,1280)!!
        assertNull(state.begin(1200,720,1280))
        val geometry=FaceGeometry.fromMesh(mesh(),720f/1280)!!
        state.complete(ticket,geometry,1100)
        assertSame(geometry,state.current(1250));assertNull(state.current(1251))
        assertNull(state.begin(1050,720,1280))
        val next=state.begin(1300,720,1280)!!
        state.complete(next,null,1350);assertNull(state.current(1350))
    }
    @Test fun snapshotCarriesEpochAndSampleTimestampUnderOneLock() {
        val state=FaceTrackingState();state.enable(true)
        val geometry=FaceGeometry.fromMesh(mesh(),1f)!!
        val ticket=state.begin(1000,256,256)!!
        state.complete(ticket,geometry,1040)
        val first=state.snapshot(1100)
        assertSame(geometry,first.geometry);assertEquals(1000L,first.sampleTimestamp)
        assertNull(state.snapshot(1251).geometry)
        state.reset()
        val second=state.snapshot(1100)
        assertNotEquals(first.epoch,second.epoch);assertNull(second.geometry)
    }
    @Test fun stopCameraChangeAndSizeChangeDiscardLateResultsWithoutQueuing() {
        val geometry=FaceGeometry.fromMesh(mesh(),1f)!!
        val state=FaceTrackingState();state.enable(true)
        val old=state.begin(1000,256,256)!!
        state.reset();assertNull(state.begin(1200,256,256))
        state.complete(old,geometry,1200);assertNull(state.current(1200))
        val current=state.begin(1300,256,256)!!
        state.enable(false);state.enable(true)
        state.complete(current,geometry,1350);assertNull(state.current(1350))
        val resized=state.begin(1400,256,256)!!
        state.begin(1450,128,256)
        state.complete(resized,geometry,1500);assertNull(state.current(1500))
    }
    @Test fun samplingIntervalIsChosenByCaller() {
        val state=FaceTrackingState();state.enable(true)
        state.complete(state.begin(1000,256,256)!!,null,1010)
        assertNull(state.begin(1099,256,256))
        assertNull(state.begin(1044,256,256,FaceTrackingState.CPU_INTERVAL_MS))
        state.complete(state.begin(1045,256,256,FaceTrackingState.CPU_INTERVAL_MS)!!,null,1050)
        assertNotNull(state.begin(1145,256,256))
    }
    @Test fun snapshotReportsLatestCompletedSampleWithOrWithoutFace() {
        val state=FaceTrackingState();state.enable(true)
        assertEquals(Long.MIN_VALUE,state.snapshot(1000).completed)
        state.complete(state.begin(1000,256,256,FaceTrackingState.FRAME_INTERVAL_MS)!!,null,1010)
        assertEquals(1000L,state.snapshot(1010).completed)
        state.complete(state.begin(1033,256,256,FaceTrackingState.FRAME_INTERVAL_MS)!!,FaceGeometry.fromMesh(mesh(),1f),1040)
        assertEquals(1033L,state.snapshot(1040).completed)
        state.reset();assertEquals(Long.MIN_VALUE,state.snapshot(1040).completed)
    }
    @Test fun resetDoesNotReuseMediaPipeVideoTimestamp() {
        val state=FaceTrackingState();state.enable(true)
        val old=state.begin(1000,256,256)!!
        state.complete(old,null,1000);state.reset()
        assertNull(state.begin(1000,256,256))
        assertNotNull(state.begin(1001,256,256))
    }
    @Test fun slowDetectionNeverPublishesStaleGeometry() {
        val state=FaceTrackingState();state.enable(true)
        val ticket=state.begin(1000,256,256)!!
        state.complete(ticket,FaceGeometry.fromMesh(mesh(),1f),1251)
        assertNull(state.current(1251))
    }
}
