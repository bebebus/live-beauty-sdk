plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

group = "io.bebebus"
version = "1.1.3"

android {
    namespace = "io.bebebus.livebeauty"
    compileSdk = 35

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
        unitTests.all {
            it.inputs.file("${projectDir}/../../reference/beauty_vectors.json")
                .withPropertyName("beautyVectors")
                .withPathSensitivity(PathSensitivity.RELATIVE)
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("com.google.mediapipe:tasks-vision:0.10.35")
    implementation("androidx.annotation:annotation:1.8.2")

    testImplementation("junit:junit:4.13.2")
}
