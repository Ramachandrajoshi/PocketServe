package com.ram.pocketserver.hardware

import android.app.ActivityManager
import android.content.Context
import android.opengl.EGL14
import android.opengl.GLES20
import android.os.Build
import android.content.pm.PackageManager
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject

data class HardwareCapability(
    val cpuAbi: String = "",
    val cpuCores: Int = 0,
    val totalRamMB: Long = 0L,
    val gpuVendor: GpuVendor = GpuVendor.OTHER,
    val gpuModel: String = "unknown",
    val supportsVulkan: Boolean = false,
    val supportsOpenCL: Boolean = false,
    val hasGpuDelegate: Boolean = false,
    val recommendedBackend: LlamaBackend = LlamaBackend.CPU_GENERIC,
    val recommendedGpuLayers: Int = 0,
)

enum class GpuVendor { QUALCOMM, ARM, IMAGINATION, OTHER }
enum class LlamaBackend {
    VULKAN_ADRENO, VULKAN_MALI, VULKAN_GENERIC,
    OPENCL, CPU_NEON, CPU_GENERIC,
}

class HardwareDetector @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun probe(): HardwareCapability {
        val cpuAbi = Build.SUPPORTED_ABIS.firstOrNull().orEmpty()
        val cpuCores = Runtime.getRuntime().availableProcessors()
        val totalRamMB = queryTotalRamMb()

        val gpuInfo = probeGpu()
        val supportsVulkan = context.packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL)
        // OpenCL detection needs native probing; keep false until explicit probe is implemented.
        val supportsOpenCL = false
        val hasGpuDelegate = supportsVulkan || supportsOpenCL
        val recommendedBackend = selectRecommendedBackend(gpuInfo.vendor, supportsVulkan, supportsOpenCL)
        val recommendedGpuLayers = recommendGpuLayers(gpuInfo.vendor, totalRamMB)

        return HardwareCapability(
            cpuAbi = cpuAbi,
            cpuCores = cpuCores,
            totalRamMB = totalRamMB,
            gpuVendor = gpuInfo.vendor,
            gpuModel = gpuInfo.model,
            supportsVulkan = supportsVulkan,
            supportsOpenCL = supportsOpenCL,
            hasGpuDelegate = hasGpuDelegate,
            recommendedBackend = recommendedBackend,
            recommendedGpuLayers = recommendedGpuLayers,
        )
    }

    private data class GpuInfo(val vendor: GpuVendor, val model: String)

    private fun queryTotalRamMb(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.totalMem / (1024L * 1024L)
    }

    private fun probeGpu(): GpuInfo {
        return runCatching {
            val eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val versions = IntArray(2)
            if (!EGL14.eglInitialize(eglDisplay, versions, 0, versions, 1)) {
                return GpuInfo(GpuVendor.OTHER, "unknown")
            }

            val configs = arrayOfNulls<android.opengl.EGLConfig>(1)
            val numConfig = IntArray(1)
            EGL14.eglChooseConfig(
                eglDisplay,
                intArrayOf(EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT, EGL14.EGL_NONE),
                0,
                configs,
                0,
                1,
                numConfig,
                0,
            )
            val eglConfig = configs.firstOrNull()
                ?: return GpuInfo(GpuVendor.OTHER, "unknown")

            val eglContext = EGL14.eglCreateContext(
                eglDisplay,
                eglConfig,
                EGL14.EGL_NO_CONTEXT,
                intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE),
                0,
            )
            val eglSurface = EGL14.eglCreatePbufferSurface(
                eglDisplay,
                eglConfig,
                intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE),
                0,
            )
            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

            val renderer = GLES20.glGetString(GLES20.GL_RENDERER).orEmpty()

            EGL14.eglDestroySurface(eglDisplay, eglSurface)
            EGL14.eglDestroyContext(eglDisplay, eglContext)
            EGL14.eglTerminate(eglDisplay)

            val vendor = when {
                renderer.contains("Adreno", ignoreCase = true) -> GpuVendor.QUALCOMM
                renderer.contains("Mali", ignoreCase = true) -> GpuVendor.ARM
                renderer.contains("PowerVR", ignoreCase = true) -> GpuVendor.IMAGINATION
                else -> GpuVendor.OTHER
            }
            GpuInfo(vendor, renderer.ifBlank { "unknown" })
        }.getOrDefault(GpuInfo(GpuVendor.OTHER, "unknown"))
    }

    private fun selectRecommendedBackend(
        vendor: GpuVendor,
        supportsVulkan: Boolean,
        supportsOpenCL: Boolean,
    ): LlamaBackend {
        return when {
            supportsVulkan && vendor == GpuVendor.QUALCOMM -> LlamaBackend.VULKAN_ADRENO
            supportsVulkan && vendor == GpuVendor.ARM -> LlamaBackend.VULKAN_MALI
            supportsVulkan -> LlamaBackend.VULKAN_GENERIC
            supportsOpenCL -> LlamaBackend.OPENCL
            Build.SUPPORTED_ABIS.any { it.contains("arm", ignoreCase = true) } -> LlamaBackend.CPU_NEON
            else -> LlamaBackend.CPU_GENERIC
        }
    }

    private fun recommendGpuLayers(vendor: GpuVendor, totalRamMB: Long): Int {
        return when {
            totalRamMB >= 8192 && (vendor == GpuVendor.QUALCOMM || vendor == GpuVendor.ARM) -> 99
            totalRamMB >= 6144 && (vendor == GpuVendor.QUALCOMM || vendor == GpuVendor.ARM) -> 32
            totalRamMB >= 4096 -> 16
            else -> 0
        }
    }
}
