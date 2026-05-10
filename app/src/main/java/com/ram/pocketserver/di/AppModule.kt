package com.ram.pocketserver.di

import com.ram.pocketserver.hardware.HardwareCapability
import com.ram.pocketserver.hardware.HardwareDetector
import com.ram.pocketserver.inference.InferenceEngine
import com.ram.pocketserver.inference.litert.LiteRTEngine
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideHardwareCapability(detector: HardwareDetector): HardwareCapability {
        return detector.probe()
    }

    @Provides
    @Singleton
    fun provideInferenceEngine(engine: LiteRTEngine): InferenceEngine = engine
}
