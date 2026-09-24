// Plugin versions are declared as buildscript classpath dependencies rather
// than with the plugins{} version syntax: this machine's Gradle dependency
// cache holds the real artifacts (verified by the monocles_chat and
// MaterialFiles builds) but not every plugin-marker artifact, and offline
// builds must not need the markers. The app module's plugins{} block applies
// the ids without versions.
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:9.0.1")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.2.10")
        classpath("org.jetbrains.kotlin:compose-compiler-gradle-plugin:2.2.10")
    }
}
