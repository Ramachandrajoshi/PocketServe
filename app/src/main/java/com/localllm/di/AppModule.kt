package com.localllm.di

import android.content.Context
import com.localllm.hardware.HardwareCapability
import com.localllm.hardware.HardwareDetector
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.hilt.android.qualifiers.ApplicationContext

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    fun provideHardwareCapability(detector: HardwareDetector): HardwareCapability {
        return detector.probe()
    }
}
