package com.ram.pocketserver.di

import android.content.Context
import com.ram.pocketserver.hardware.HardwareCapability
import com.ram.pocketserver.hardware.HardwareDetector
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
}
