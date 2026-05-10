package com.ram.pocketserver.di

import android.content.Context
import com.ram.pocketserver.hardware.HardwareCapability
import com.ram.pocketserver.hardware.HardwareDetector
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
