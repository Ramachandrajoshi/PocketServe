package com.ram.pocketserver.hardware

import javax.inject.Inject
import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext

data class HardwareCapability(
    val hasGpuDelegate: Boolean
)

class HardwareDetector @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun probe(): HardwareCapability {
        return HardwareCapability(hasGpuDelegate = false)
    }
}
