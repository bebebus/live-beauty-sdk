plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("maven-publish")
}

group = "io.bebebus"
version = "3.9.0"

android {
    namespace = "io.bebebus.livebeauty"
    compileSdk = 35
    val detectedNdkVersion = System.getenv("ANDROID_NDK_VERSION")
        ?: listOfNotNull(System.getenv("ANDROID_HOME"), System.getenv("ANDROID_SDK_ROOT"))
            .map { file("$it/ndk") }
            .firstOrNull { it.isDirectory }
            ?.listFiles()
            ?.filter { it.isDirectory && file("${it.absolutePath}/source.properties").isFile }
            ?.map { it.name }
            ?.sortedDescending()
            ?.firstOrNull()
    if (detectedNdkVersion != null) {
        ndkVersion = detectedNdkVersion
    }

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                arguments(
                    "-DLIVEBEAUTY_BUILD_TESTS=OFF",
                    "-DLIVEBEAUTY_ENABLE_GLES=ON",
                    "-DLIVEBEAUTY_BUILD_JNI=ON",
                )
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("../../core/CMakeLists.txt")
        }
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

    publishing {
        singleVariant("release") {
            withSourcesJar()
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

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                groupId = "io.bebebus"
                artifactId = "live-beauty"
                version = project.version.toString()
                pom {
                    name.set("Bebebus Live Beauty SDK")
                    description.set("Unified C++17 math & RHI render core (GLES/Metal) for live streaming beauty, virtual background and 2D AR stickers.")
                    url.set("https://github.com/bebebus/live-beauty-sdk")
                    licenses {
                        license {
                            name.set("The Apache License, Version 2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                        }
                    }
                }
            }
        }
    }
}
