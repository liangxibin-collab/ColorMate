plugins {
    id("com.android.application")
    id("com.chaquo.python")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.colormate.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.colormate.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        ndk {
            // Only target arm64-v8a for now; x86_64 causes mergeDebugNativeLibs
            // failures with Chaquopy native library transforms on CI runners.
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    packaging {
        resources {
            excludes += setOf(
                "META-INF/LICENSE*",
                "META-INF/NOTICE*",
                "LICENSE.txt",
                "LICENSE"
            )
        }
    }
}

// Chaquopy 17+ DSL uses "chaquopy" block (not "python" block, not inside android)
chaquopy {
    defaultConfig {
        version = "3.11"
        pip {
            // Minimal deps: opencv handles color analysis, numpy for arrays,
            // Pillow for image I/O. scipy and scikit-learn removed to reduce
            // build time and CI memory pressure.
            install("opencv-python")
            install("numpy")
            install("Pillow")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.documentfile:documentfile:1.0.1")
    implementation("com.google.android.material:material:1.12.0")
}
