# LocalLLM Studio — Android App
## Technical Documentation & Architecture Guide

**Version:** 1.0  
**Platform:** Android 9.0 (API 28) and above  
**Target API:** 35  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [App Architecture](#2-app-architecture)
3. [Module Structure](#3-module-structure)
4. [Inference Framework Integration](#4-inference-framework-integration)
5. [GPU / NPU Detection & Binary Selection](#5-gpu--npu-detection--binary-selection)
6. [OpenAI-Compatible Server Layer](#6-openai-compatible-server-layer)
7. [Model Management & Download System](#7-model-management--download-system)
8. [Server Configuration & Advanced Settings](#8-server-configuration--advanced-settings)
9. [Chat UI & Testing Interface](#9-chat-ui--testing-interface)
10. [Benchmark System](#10-benchmark-system)
11. [Wake Lock & Background Service](#11-wake-lock--background-service)
12. [Dependencies & Gradle Configuration](#12-dependencies--gradle-configuration)
13. [Recommended Models & Best Settings Guide](#13-recommended-models--best-settings-guide)
14. [Security & Permissions](#14-security--permissions)
15. [Build Variants & Flavors](#15-build-variants--flavors)

---

## 1. Project Overview

**LocalLLM Studio** is a fully self-contained Android application that turns a smartphone into a local LLM inference server. Users can:

- Choose between **MNN**, **llama.cpp**, and **LiteRT-LM** inference backends.
- Download quantized GGUF / TFLite / MNN models directly in-app.
- Start an **OpenAI-compatible HTTP server** (equivalent to vLLM's full endpoint set) on the device.
- Chat with the model locally through a polished in-app UI.
- Run **benchmarks** to measure tokens/sec, memory, and thermal performance.
- Keep the server alive with an optional **wake lock + foreground service** even when the screen is off.
- Receive guided recommendations for the best models and settings for the detected hardware.

---

## 2. App Architecture

The app follows **Clean Architecture** with **MVVM** presentation layer and a unidirectional data flow pattern.

```
┌─────────────────────────────────────────────────────┐
│                  Presentation Layer                  │
│   (Jetpack Compose UI, ViewModels, UiState flows)   │
├─────────────────────────────────────────────────────┤
│                   Domain Layer                       │
│   (UseCases, Repository Interfaces, Domain Models)  │
├─────────────────────────────────────────────────────┤
│                    Data Layer                        │
│  (Room DB, DataStore, Model Downloader, Inference)  │
├─────────────────────────────────────────────────────┤
│               Native / JNI Layer                     │
│   (llama.cpp JNI, MNN JNI, LiteRT JNI bridges)      │
└─────────────────────────────────────────────────────┘
```

### Core Design Principles

- **Single Activity** with Compose Navigation.
- **Hilt** for dependency injection throughout all layers.
- **Kotlin Coroutines + Flow** for async inference streaming.
- **WorkManager** for download tasks (survives process death).
- **Foreground Service** for the server daemon, keeping it alive under Doze mode.

---

## 3. Module Structure

```
LocalLLMStudio/
├── app/                          # Main application module
│   ├── src/main/
│   │   ├── java/com/localllm/
│   │   │   ├── MainActivity.kt
│   │   │   ├── LocalLLMApp.kt    # Hilt Application class
│   │   │   └── di/               # Hilt modules
│   │   └── res/
├── core/
│   ├── core-common/              # Extensions, utilities, base classes
│   ├── core-data/                # Repository implementations, Room, DataStore
│   ├── core-domain/              # UseCases, domain models, interfaces
│   ├── core-network/             # OkHttp, Retrofit, Ktor server setup
│   └── core-ui/                  # Shared Compose components, theme
├── feature/
│   ├── feature-home/             # Dashboard / hardware summary screen
│   ├── feature-framework/        # Framework selector screen
│   ├── feature-models/           # Model browser, download manager
│   ├── feature-server/           # Server controls, logs, endpoint explorer
│   ├── feature-chat/             # Chat UI and streaming message view
│   ├── feature-settings/         # Advanced server config, preferences
│   └── feature-benchmark/        # Benchmark runner and results viewer
├── inference/
│   ├── inference-api/            # InferenceEngine interface (common contract)
│   ├── inference-mnn/            # MNN backend (JNI + Kotlin wrapper)
│   ├── inference-llamacpp/      # llama.cpp backend (JNI + Kotlin wrapper)
│   └── inference-litert/         # LiteRT-LM backend (JNI + Kotlin wrapper)
├── server/
│   ├── server-ktor/              # Ktor HTTP server, OpenAI route definitions
│   └── server-service/           # Android ForegroundService + WakeLock mgr
└── hardware/
    └── hardware-detect/          # GPU/NPU detection, capability probing
```

---

## 4. Inference Framework Integration

### 4.1 Unified InferenceEngine Interface

All three backends implement the same contract, making them interchangeable at runtime.

```kotlin
// inference-api/src/main/java/com/localllm/inference/InferenceEngine.kt

interface InferenceEngine {
    val name: String
    val supportedFormats: List<ModelFormat>  // GGUF, MNN, TFLITE

    suspend fun loadModel(config: ModelLoadConfig): Result<Unit>
    suspend fun unloadModel()

    fun generate(
        messages: List<ChatMessage>,
        params: GenerationParams,
    ): Flow<TokenChunk>

    suspend fun tokenize(text: String): List<Int>
    suspend fun detokenize(tokens: List<Int>): String
    suspend fun getModelInfo(): ModelInfo
    fun isModelLoaded(): Boolean
}

data class GenerationParams(
    val maxTokens: Int = 512,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val repeatPenalty: Float = 1.1f,
    val seed: Long = -1L,
    val stop: List<String> = emptyList(),
    val stream: Boolean = true,
    val minP: Float = 0.05f,
    val tfsZ: Float = 1.0f,
    val typicalP: Float = 1.0f,
    val mirostatMode: Int = 0,
    val mirostatTau: Float = 5.0f,
    val mirostatEta: Float = 0.1f,
    val grammar: String? = null,           // GBNF grammar string
    val jsonSchema: String? = null,        // JSON schema for constrained output
)
```

### 4.2 llama.cpp Backend

llama.cpp uses its official Android JNI bindings. The app packages multiple ABI-specific `.so` files and selects the right one at load time based on hardware detection.

```kotlin
// inference-llamacpp/src/main/java/com/localllm/inference/llamacpp/LlamaCppEngine.kt

class LlamaCppEngine @Inject constructor(
    private val hardwareCapability: HardwareCapability,
    private val binarySelector: LlamaBinarySelector,
) : InferenceEngine {

    private external fun nativeLoadModel(
        modelPath: String,
        nCtx: Int,
        nThreads: Int,
        nGpuLayers: Int,
        useMmap: Boolean,
        useMlock: Boolean,
        flashAttention: Boolean,
    ): Long  // returns context pointer

    private external fun nativeGenerate(
        ctx: Long,
        prompt: String,
        params: ByteArray,  // serialized proto
        callbackId: Int,
    ): Int

    private external fun nativeGetKVCacheUsage(ctx: Long): Float
    private external fun nativeSetLora(ctx: Long, loraPath: String, scale: Float)

    companion object {
        init {
            // Loaded conditionally based on hardware; see Section 5
            System.loadLibrary("llama_jni")
        }
    }
}
```

**CMakeLists integration for llama.cpp:**
```cmake
# inference-llamacpp/src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.22.1)
project(llama_jni)

set(LLAMA_METAL OFF)
set(LLAMA_VULKAN ON)
set(LLAMA_OPENCL ON)
set(LLAMA_LLAMAFILE OFF)

add_subdirectory(llama.cpp)

add_library(llama_jni SHARED llama_jni.cpp)
target_link_libraries(llama_jni llama ggml android log)
```

### 4.3 MNN Backend

MNN (Mobile Neural Network) is Alibaba's inference framework, optimized for ARM CPUs and their custom NPU backends.

```kotlin
// inference-mnn/src/main/java/com/localllm/inference/mnn/MNNEngine.kt

class MNNEngine @Inject constructor(
    private val context: Context,
) : InferenceEngine {
    override val name = "MNN"
    override val supportedFormats = listOf(ModelFormat.MNN, ModelFormat.GGUF)

    private var mnnLLM: MNNLLMSession? = null

    override suspend fun loadModel(config: ModelLoadConfig): Result<Unit> {
        return withContext(Dispatchers.IO) {
            runCatching {
                val configJson = buildMNNConfig(config)
                mnnLLM = MNNLLMSession.create(config.modelPath, configJson)
            }
        }
    }

    private fun buildMNNConfig(config: ModelLoadConfig): String {
        return JSONObject().apply {
            put("backend_type", config.backendType.mnnValue)  // CPU=0, OPENCL=3, VULKAN=7
            put("thread_num", config.threads)
            put("precision", config.precision.mnnValue)       // HIGH, LOW, NORMAL
            put("memory", config.memoryMode.mnnValue)         // HIGH, LOW, NORMAL
            put("kv_cache_size_mb", config.kvCacheSizeMB)
            put("prefill_split", config.prefillChunkSize)
        }.toString()
    }

    override fun generate(messages: List<ChatMessage>, params: GenerationParams): Flow<TokenChunk> =
        callbackFlow {
            val prompt = messages.toPromptString()
            mnnLLM?.generateAsync(prompt, params.maxTokens) { token, done ->
                trySend(TokenChunk(token, done))
                if (done) close()
            } ?: close(IllegalStateException("Model not loaded"))
            awaitClose()
        }
}
```

### 4.4 LiteRT-LM Backend

LiteRT-LM (formerly TFLite with LLM extensions from Google) runs `.task` or `.tflite` models and leverages GPU delegate on Adreno / Mali.

```kotlin
// inference-litert/src/main/java/com/localllm/inference/litert/LiteRTEngine.kt

class LiteRTEngine @Inject constructor(
    private val context: Context,
    private val hardwareCapability: HardwareCapability,
) : InferenceEngine {
    override val name = "LiteRT-LM"
    override val supportedFormats = listOf(ModelFormat.TFLITE_TASK, ModelFormat.SAFETENSORS)

    private var llmInference: LlmInference? = null

    override suspend fun loadModel(config: ModelLoadConfig): Result<Unit> {
        return withContext(Dispatchers.IO) {
            runCatching {
                val options = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(config.modelPath)
                    .setMaxTokens(config.contextLength)
                    .setTopK(config.topK)
                    .setTemperature(config.temperature)
                    .setRandomSeed(config.seed)
                    .apply {
                        if (hardwareCapability.hasGpuDelegate) {
                            setPreferredBackend(LlmInference.Backend.GPU)
                        }
                    }
                    .build()
                llmInference = LlmInference.createFromOptions(context, options)
            }
        }
    }

    override fun generate(messages: List<ChatMessage>, params: GenerationParams): Flow<TokenChunk> =
        callbackFlow {
            val prompt = messages.toGemmaPrompt()
            llmInference?.generateResponseAsync(prompt) { partial, done ->
                trySend(TokenChunk(partial ?: "", done))
                if (done) close()
            } ?: close(IllegalStateException("Model not loaded"))
            awaitClose()
        }
}
```

### 4.5 Framework Selector & Dynamic Switching

```kotlin
// core-domain/src/main/java/com/localllm/domain/usecase/SelectFrameworkUseCase.kt

class SelectFrameworkUseCase @Inject constructor(
    private val settingsRepository: SettingsRepository,
    @MNNEngine private val mnnEngine: InferenceEngine,
    @LlamaCppEngine private val llamaCppEngine: InferenceEngine,
    @LiteRTEngine private val liteRTEngine: InferenceEngine,
) {
    fun getEngine(framework: Framework): InferenceEngine = when (framework) {
        Framework.MNN       -> mnnEngine
        Framework.LLAMACPP -> llamaCppEngine
        Framework.LITERT    -> liteRTEngine
    }
}
```

---

## 5. GPU / NPU Detection & Binary Selection

This is one of the most critical sections. The app probes hardware capabilities at launch and selects the optimal native binary and backend configuration automatically.

### 5.1 Hardware Detection Flow

```
App Launch
    │
    ▼
HardwareDetector.probe()
    ├── CPU Info  (ABI, core count, frequency, ARM features)
    ├── GPU Info  (OpenGL ES renderer string → vendor/model)
    ├── Vulkan    (vkEnumeratePhysicalDevices → device name, driver)
    ├── OpenCL    (clGetPlatformInfo / clGetDeviceInfo)
    ├── NNAPI     (NeuralNetworks API version, accelerator list)
    └── HTA/DSP   (Qualcomm QNN / MediaTek APU / Samsung NPU probes)
    │
    ▼
HardwareCapability object
    │
    ▼
BinarySelector.select(capability) → LlamaBackend enum
    │
    ├── VULKAN_ADRENO   → libllama_vulkan_adreno.so
    ├── VULKAN_MALI     → libllama_vulkan_mali.so
    ├── OPENCL          → libllama_opencl.so
    ├── NNAPI           → libllama_nnapi.so
    └── CPU_NEON        → libllama_cpu_neon.so  (fallback)
```

### 5.2 HardwareDetector Implementation

```kotlin
// hardware-detect/src/main/java/com/localllm/hardware/HardwareDetector.kt

data class HardwareCapability(
    val cpuAbi: String,               // arm64-v8a, x86_64
    val cpuCores: Int,
    val cpuBigCores: Int,             // P-cores / big cluster count
    val cpuFeatures: Set<CpuFeature>, // SVE, SVE2, DOTPROD, FP16, BF16
    val totalRamMB: Long,
    val gpuVendor: GpuVendor,         // QUALCOMM, ARM, IMAGINATION, OTHER
    val gpuModel: String,             // "Adreno 750", "Mali-G715"
    val gpuGeneration: Int,           // Adreno 700-series → 700
    val supportsVulkan: Boolean,
    val vulkanVersion: String?,        // "1.3.0"
    val supportsOpenCL: Boolean,
    val openCLVersion: String?,
    val supportsNNAPI: Boolean,
    val nnapiVersion: Int,
    val hasQualcommHTA: Boolean,       // Snapdragon HTA / Hexagon DSP
    val hasMediaTekAPU: Boolean,
    val hasSamsungNPU: Boolean,
    val hasGpuDelegate: Boolean,       // LiteRT GPU delegate viable
    val recommendedBackend: LlamaBackend,
    val recommendedGpuLayers: Int,
)

enum class GpuVendor { QUALCOMM, ARM, IMAGINATION, APPLE, OTHER }
enum class LlamaBackend {
    VULKAN_ADRENO, VULKAN_MALI, VULKAN_GENERIC,
    OPENCL, NNAPI, CPU_NEON, CPU_GENERIC
}
enum class CpuFeature { SVE, SVE2, DOTPROD, FP16, BF16, I8MM }

class HardwareDetector @Inject constructor(
    private val context: Context,
) {
    fun probe(): HardwareCapability {
        val cpuInfo = probeCPU()
        val gpuInfo = probeGPU()
        val vulkan  = probeVulkan()
        val openCL  = probeOpenCL()
        val nnapi   = probeNNAPI()
        val npu     = probeNPU(gpuInfo.vendor)

        val backend = selectOptimalBackend(gpuInfo, vulkan, openCL, nnapi, npu)
        val gpuLayers = recommendGpuLayers(gpuInfo, cpuInfo.totalRamMB)

        return HardwareCapability(
            cpuAbi             = Build.SUPPORTED_ABIS.first(),
            cpuCores           = Runtime.getRuntime().availableProcessors(),
            cpuBigCores        = cpuInfo.bigCores,
            cpuFeatures        = cpuInfo.features,
            totalRamMB         = cpuInfo.totalRamMB,
            gpuVendor          = gpuInfo.vendor,
            gpuModel           = gpuInfo.model,
            gpuGeneration      = gpuInfo.generation,
            supportsVulkan     = vulkan.supported,
            vulkanVersion      = vulkan.version,
            supportsOpenCL     = openCL.supported,
            openCLVersion      = openCL.version,
            supportsNNAPI      = nnapi.supported,
            nnapiVersion       = nnapi.version,
            hasQualcommHTA     = npu.qualcommHTA,
            hasMediaTekAPU     = npu.mediaTekAPU,
            hasSamsungNPU      = npu.samsungNPU,
            hasGpuDelegate     = vulkan.supported || openCL.supported,
            recommendedBackend = backend,
            recommendedGpuLayers = gpuLayers,
        )
    }

    private fun probeGPU(): GpuInfo {
        // Create minimal EGL context, query GL_RENDERER string
        val eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        EGL14.eglInitialize(eglDisplay, IntArray(1), 0, IntArray(1), 0)
        val configs = arrayOfNulls<EGLConfig>(1)
        EGL14.eglChooseConfig(eglDisplay, intArrayOf(EGL14.EGL_NONE), 0, configs, 0, 1, IntArray(1), 0)
        val surface = EGL14.eglCreatePbufferSurface(eglDisplay, configs[0], intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
        val ctx = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0)
        EGL14.eglMakeCurrent(eglDisplay, surface, surface, ctx)

        val renderer = GLES20.glGetString(GLES20.GL_RENDERER) ?: ""
        val vendor   = GLES20.glGetString(GLES20.GL_VENDOR)   ?: ""

        EGL14.eglDestroyContext(eglDisplay, ctx)
        EGL14.eglDestroySurface(eglDisplay, surface)
        EGL14.eglTerminate(eglDisplay)

        return parseGpuInfo(renderer, vendor)
    }

    private fun parseGpuInfo(renderer: String, vendor: String): GpuInfo {
        return when {
            renderer.contains("Adreno", ignoreCase = true) -> {
                val gen = Regex("Adreno \\(TM\\) (\\d+)").find(renderer)
                    ?.groupValues?.get(1)?.take(1)?.toIntOrNull()?.times(100) ?: 600
                GpuInfo(GpuVendor.QUALCOMM, renderer, gen)
            }
            renderer.contains("Mali", ignoreCase = true) -> {
                val gen = Regex("Mali-G(\\d+)").find(renderer)
                    ?.groupValues?.get(1)?.toIntOrNull() ?: 0
                GpuInfo(GpuVendor.ARM, renderer, gen)
            }
            renderer.contains("PowerVR", ignoreCase = true) ->
                GpuInfo(GpuVendor.IMAGINATION, renderer, 0)
            else -> GpuInfo(GpuVendor.OTHER, renderer, 0)
        }
    }

    private fun probeVulkan(): VulkanInfo {
        return try {
            val supported = context.packageManager
                .hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_COMPUTE, 0)
            val version = if (supported) probeVulkanVersionNative() else null
            VulkanInfo(supported, version)
        } catch (e: Exception) { VulkanInfo(false, null) }
    }

    private external fun probeVulkanVersionNative(): String
    private external fun probeOpenCLNative(): OpenCLInfo
    private external fun probeQualcommHTANative(): Boolean

    private fun probeNNAPI(): NNAPIInfo {
        val version = NeuralNetworks.getRuntimeVersion() ?: return NNAPIInfo(false, 0)
        val level = version.split(".").firstOrNull()?.toIntOrNull() ?: 0
        return NNAPIInfo(level >= 1, level)
    }

    private fun selectOptimalBackend(
        gpu: GpuInfo, vulkan: VulkanInfo, openCL: OpenCLInfo,
        nnapi: NNAPIInfo, npu: NPUInfo,
    ): LlamaBackend = when {
        vulkan.supported && gpu.vendor == GpuVendor.QUALCOMM && gpu.generation >= 600 ->
            LlamaBackend.VULKAN_ADRENO
        vulkan.supported && gpu.vendor == GpuVendor.ARM ->
            LlamaBackend.VULKAN_MALI
        vulkan.supported ->
            LlamaBackend.VULKAN_GENERIC
        openCL.supported ->
            LlamaBackend.OPENCL
        nnapi.supported && nnapi.version >= 3 ->
            LlamaBackend.NNAPI
        else ->
            LlamaBackend.CPU_NEON
    }

    private fun recommendGpuLayers(gpu: GpuInfo, ramMB: Long): Int = when {
        gpu.vendor == GpuVendor.QUALCOMM && gpu.generation >= 700 && ramMB >= 8192 -> 99
        gpu.vendor == GpuVendor.QUALCOMM && gpu.generation >= 600 && ramMB >= 6144 -> 32
        gpu.vendor == GpuVendor.ARM && gpu.generation >= 715 && ramMB >= 8192 -> 99
        gpu.vendor == GpuVendor.ARM && ramMB >= 6144 -> 28
        ramMB >= 8192 -> 16
        else -> 0
    }
}
```

### 5.3 Dynamic .so Loading

The app ships multiple `.so` variants for llama.cpp under `jniLibs`. A custom loader extracts and dlopen()s the right one at runtime:

```kotlin
// hardware-detect/src/main/java/com/localllm/hardware/LlamaBinarySelector.kt

class LlamaBinarySelector @Inject constructor(
    private val context: Context,
) {
    private val soMap = mapOf(
        LlamaBackend.VULKAN_ADRENO  to "libllama_vulkan_adreno.so",
        LlamaBackend.VULKAN_MALI    to "libllama_vulkan_mali.so",
        LlamaBackend.VULKAN_GENERIC to "libllama_vulkan.so",
        LlamaBackend.OPENCL         to "libllama_opencl.so",
        LlamaBackend.NNAPI          to "libllama_nnapi.so",
        LlamaBackend.CPU_NEON       to "libllama_cpu_neon.so",
        LlamaBackend.CPU_GENERIC    to "libllama_cpu.so",
    )

    fun loadLibraryForBackend(backend: LlamaBackend) {
        val soName = soMap[backend] ?: "libllama_cpu_neon.so"
        // Extract from APK assets if not already in nativeLibsDir
        val libFile = File(context.applicationInfo.nativeLibraryDir, soName)
        if (!libFile.exists()) extractFromAssets(soName, libFile)
        System.load(libFile.absolutePath)
    }

    private fun extractFromAssets(name: String, dest: File) {
        context.assets.open("libs/${Build.SUPPORTED_ABIS.first()}/$name")
            .use { input -> dest.outputStream().use { input.copyTo(it) } }
    }
}
```

### 5.4 ABI Layout in APK

```
app/src/main/
├── jniLibs/
│   └── arm64-v8a/
│       ├── libllama_vulkan_adreno.so   # Built with -DGGML_VULKAN + Adreno tuning
│       ├── libllama_vulkan_mali.so     # Built with -DGGML_VULKAN + Mali tuning
│       ├── libllama_vulkan.so          # Generic Vulkan
│       ├── libllama_opencl.so          # Built with -DGGML_OPENCL
│       ├── libllama_nnapi.so           # Built with -DGGML_NNAPI
│       └── libllama_cpu_neon.so        # CPU-only with NEON intrinsics
└── assets/
    └── libs/
        └── arm64-v8a/                  # Overflow .so files for dynamic extraction
```

---

## 6. OpenAI-Compatible Server Layer

The app runs a **Ktor** HTTP server on a user-configured port (default: 8080). All endpoints mirror the OpenAI API spec and vLLM's extended capabilities.

### 6.1 Endpoint List

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List loaded models |
| GET | `/v1/models/{model_id}` | Retrieve model info |
| POST | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| POST | `/v1/completions` | Text completions (legacy) |
| POST | `/v1/embeddings` | Text embeddings |
| POST | `/v1/tokenize` | Tokenize text (llama.cpp extension) |
| POST | `/v1/detokenize` | Detokenize token IDs |
| GET | `/v1/slots` | KV cache slot status (llama.cpp extension) |
| POST | `/v1/slots/{id}/save` | Save KV cache slot |
| POST | `/v1/slots/{id}/restore` | Restore KV cache slot |
| GET | `/health` | Server health + model status |
| GET | `/metrics` | Prometheus-format metrics |
| GET | `/v1/queue` | Request queue status |
| POST | `/v1/lora/load` | Load LoRA adapter |
| DELETE | `/v1/lora/{id}` | Unload LoRA adapter |
| GET | `/docs` | Swagger UI (debug builds only) |

### 6.2 Ktor Server Configuration

```kotlin
// server-ktor/src/main/java/com/localllm/server/LLMServer.kt

class LLMServer @Inject constructor(
    private val inferenceEngine: InferenceEngine,
    private val serverConfig: ServerConfig,
) {
    private var server: ApplicationEngine? = null

    fun start() {
        server = embeddedServer(Netty, port = serverConfig.port, host = serverConfig.host) {
            install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) }
            install(CORS) {
                allowMethod(HttpMethod.Options)
                allowMethod(HttpMethod.Post)
                allowMethod(HttpMethod.Get)
                allowMethod(HttpMethod.Delete)
                allowHeader(HttpHeaders.Authorization)
                allowHeader(HttpHeaders.ContentType)
                anyHost()
            }
            install(CallLogging) { level = Level.INFO }
            install(RateLimit) {
                register {
                    rateLimiter(limit = serverConfig.rateLimit, refillPeriod = 1.minutes)
                }
            }
            if (serverConfig.authEnabled) {
                install(Authentication) {
                    bearer("api-key") {
                        authenticate { credential ->
                            if (serverConfig.apiKeys.contains(credential.token))
                                UserIdPrincipal(credential.token) else null
                        }
                    }
                }
            }
            routing {
                modelsRoutes(inferenceEngine)
                chatCompletionsRoutes(inferenceEngine, serverConfig)
                completionsRoutes(inferenceEngine, serverConfig)
                embeddingsRoutes(inferenceEngine)
                tokenizerRoutes(inferenceEngine)
                slotsRoutes(inferenceEngine)
                healthRoutes(inferenceEngine)
                metricsRoutes()
                loraRoutes(inferenceEngine)
            }
        }.start(wait = false)
    }

    fun stop() { server?.stop(1000, 5000) }
}
```

### 6.3 Chat Completions with SSE Streaming

```kotlin
// server-ktor/src/main/java/com/localllm/server/routes/ChatRoutes.kt

fun Route.chatCompletionsRoutes(engine: InferenceEngine, config: ServerConfig) {
    post("/v1/chat/completions") {
        val request = call.receive<ChatCompletionRequest>()

        if (request.stream == true) {
            call.response.header(HttpHeaders.ContentType, "text/event-stream")
            call.response.header("Cache-Control", "no-cache")
            call.response.header("X-Accel-Buffering", "no")

            call.respondTextStream {
                engine.generate(request.messages, request.toGenerationParams())
                    .collect { chunk ->
                        val delta = ChatCompletionChunk(
                            id      = "chatcmpl-${UUID.randomUUID()}",
                            model   = engine.getModelInfo().id,
                            choices = listOf(
                                Choice(index = 0, delta = Delta(content = chunk.text),
                                    finishReason = if (chunk.done) "stop" else null)
                            )
                        )
                        emit("data: ${Json.encodeToString(delta)}\n\n")
                    }
                emit("data: [DONE]\n\n")
            }
        } else {
            val sb = StringBuilder()
            var promptTokens = 0; var completionTokens = 0
            engine.generate(request.messages, request.toGenerationParams()).collect { chunk ->
                sb.append(chunk.text)
                completionTokens++
            }
            call.respond(ChatCompletionResponse(
                choices = listOf(Choice(message = Message(role = "assistant", content = sb.toString()), finishReason = "stop")),
                usage = Usage(promptTokens, completionTokens, promptTokens + completionTokens),
            ))
        }
    }
}
```

---

## 7. Model Management & Download System

### 7.1 Model Registry

The app ships a curated `models.json` (bundled in assets, refreshed from a CDN) listing recommended models per framework with metadata:

```json
{
  "models": [
    {
      "id": "gemma-3-4b-it-q4",
      "name": "Gemma 3 4B Instruct (Q4_K_M)",
      "framework": ["llamacpp", "mnn"],
      "format": "gguf",
      "size_gb": 2.5,
      "context_length": 8192,
      "quant": "Q4_K_M",
      "recommended_gpu_layers": 32,
      "min_ram_gb": 4,
      "hf_repo": "google/gemma-3-4b-it-GGUF",
      "hf_file": "gemma-3-4b-it-q4_k_m.gguf",
      "badge": "RECOMMENDED",
      "benchmark_tps": {"sd_865": 18, "sd_8g2": 28, "sd_8g3": 42},
      "chat_template": "gemma",
      "description": "Excellent balance of quality and speed on mid-range devices."
    }
  ]
}
```

### 7.2 Download Manager

```kotlin
// core-data/src/main/java/com/localllm/data/download/ModelDownloadManager.kt

class ModelDownloadManager @Inject constructor(
    private val workManager: WorkManager,
    private val modelDao: ModelDao,
) {
    fun enqueueDownload(model: ModelEntry): UUID {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .setRequiresStorageNotLow(true)
            .build()

        val request = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setConstraints(constraints)
            .setInputData(workDataOf(
                "model_id"  to model.id,
                "hf_repo"   to model.hfRepo,
                "hf_file"   to model.hfFile,
                "dest_path" to getModelStoragePath(model),
            ))
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .addTag("model_download_${model.id}")
            .build()

        workManager.enqueueUniqueWork("download_${model.id}", ExistingWorkPolicy.KEEP, request)
        return request.id
    }

    // Download progress as Flow
    fun observeDownload(workId: UUID): Flow<DownloadState> =
        workManager.getWorkInfoByIdFlow(workId).map { info ->
            when (info?.state) {
                WorkInfo.State.RUNNING   -> DownloadState.Downloading(
                    info.progress.getFloat("progress", 0f),
                    info.progress.getLong("bytes_downloaded", 0),
                    info.progress.getLong("total_bytes", 0),
                )
                WorkInfo.State.SUCCEEDED -> DownloadState.Completed
                WorkInfo.State.FAILED    -> DownloadState.Failed(info.outputData.getString("error") ?: "Unknown")
                else                     -> DownloadState.Idle
            }
        }

    private fun getModelStoragePath(model: ModelEntry): String =
        File(context.getExternalFilesDir("models"), "${model.id}.${model.format}").absolutePath
}
```

---

## 8. Server Configuration & Advanced Settings

The Settings screen exposes full control over every inference and server parameter.

### 8.1 ServerConfig Data Class

```kotlin
data class ServerConfig(
    // Network
    val host: String = "0.0.0.0",
    val port: Int = 8080,
    val authEnabled: Boolean = false,
    val apiKeys: List<String> = emptyList(),
    val corsEnabled: Boolean = true,
    val rateLimit: Int = 60,              // requests/minute
    val maxConcurrentRequests: Int = 4,

    // Model loading
    val contextLength: Int = 4096,
    val gpuLayers: Int = -1,              // -1 = auto
    val threads: Int = -1,               // -1 = auto (use big cores)
    val batchSize: Int = 512,
    val ubatchSize: Int = 128,           // micro-batch for prompt processing
    val useMmap: Boolean = true,
    val useMlock: Boolean = false,
    val flashAttention: Boolean = true,
    val kvCacheType: KVCacheType = KVCacheType.F16,  // F16, Q8_0, Q4_0
    val ropeScaling: RopeScaling = RopeScaling.NONE, // NONE, LINEAR, YARN
    val ropeFreqBase: Float = 0f,        // 0 = default
    val ropeFreqScale: Float = 0f,

    // Sampling defaults (overridable per request)
    val defaultTemperature: Float = 0.7f,
    val defaultTopP: Float = 0.9f,
    val defaultTopK: Int = 40,
    val defaultMaxTokens: Int = 512,
    val defaultRepeatPenalty: Float = 1.1f,

    // Parallel / continuous batching
    val parallelSlots: Int = 1,
    val continuousBatching: Boolean = false,

    // LoRA
    val loraAdapters: List<LoraConfig> = emptyList(),

    // Logging
    val logLevel: LogLevel = LogLevel.INFO,
    val logRequests: Boolean = true,
    val metricsEnabled: Boolean = true,

    // Wake lock
    val wakeLockEnabled: Boolean = false,
    val wifiLockEnabled: Boolean = false,
)
```

### 8.2 Settings UI Sections

The Settings screen (Jetpack Compose) organizes these into expandable sections:

- **Network** — Host, port, authentication, CORS, rate limiting
- **Model Loading** — Context window, GPU layers slider, thread count, memory mapping
- **Performance** — Batch size, flash attention, KV cache type, continuous batching
- **Sampling Defaults** — Temperature, Top-P, Top-K, penalty defaults
- **Context Extension** — RoPE scaling type and frequency settings
- **LoRA Adapters** — Add/remove LoRA files with scale control
- **Background & Power** — Wake lock, Wi-Fi lock, notification settings
- **Logging & Debug** — Log verbosity, request logging, Prometheus metrics

---

## 9. Chat UI & Testing Interface

### 9.1 Features

- Markdown rendering (code blocks, tables, bold, lists) via `compose-markdown`.
- Streaming token display with animated cursor.
- Conversation history with session save/load (Room).
- System prompt editor with templates (assistant, code helper, instruct).
- Parameter override panel per conversation (temp, top-p, max tokens).
- Token count display and context usage progress bar.
- Copy / regenerate / edit message actions.
- Multi-turn conversation tree (branch and explore alternate responses).
- Syntax-highlighted code blocks with copy button.
- Export conversation as Markdown, JSON, or plain text.

### 9.2 ChatViewModel

```kotlin
// feature-chat/src/main/java/com/localllm/chat/ChatViewModel.kt

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val inferenceEngine: InferenceEngine,
    private val conversationRepository: ConversationRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    fun sendMessage(userText: String, params: GenerationParams) {
        viewModelScope.launch {
            val updatedHistory = _uiState.value.messages + ChatMessage("user", userText)
            _uiState.update { it.copy(messages = updatedHistory, isGenerating = true, streamingText = "") }

            val sb = StringBuilder()
            inferenceEngine.generate(updatedHistory, params)
                .onEach { chunk ->
                    sb.append(chunk.text)
                    _uiState.update { it.copy(streamingText = sb.toString()) }
                }
                .onCompletion { error ->
                    val finalMsg = ChatMessage("assistant", sb.toString())
                    _uiState.update { state ->
                        state.copy(
                            messages      = state.messages + finalMsg,
                            streamingText = "",
                            isGenerating  = false,
                            error         = error?.message,
                        )
                    }
                    conversationRepository.saveMessage(finalMsg)
                }
                .catch { e -> _uiState.update { it.copy(isGenerating = false, error = e.message) } }
                .collect()
        }
    }

    fun stopGeneration() { /* cancel the collection scope */ }
    fun clearConversation() { _uiState.update { ChatUiState() } }
}
```

---

## 10. Benchmark System

The Benchmark module provides a structured way to measure and compare inference performance across frameworks, models, and hardware configurations.

### 10.1 Benchmark Metrics

| Metric | Description |
|--------|-------------|
| **Prompt Tokens/sec** (PP-TPS) | Prompt processing throughput |
| **Generation Tokens/sec** (TG-TPS) | Token generation throughput |
| **Time to First Token** (TTFT) | Latency before first token appears (ms) |
| **Memory Used** | Peak RSS + GPU VRAM allocated (MB) |
| **Thermal State** | CPU/GPU temperature at start and end |
| **Battery Draw** | mA drawn during inference (if accessible) |
| **KV Cache Efficiency** | Tokens reused per second via KV caching |

### 10.2 Benchmark Runner

```kotlin
// feature-benchmark/src/main/java/com/localllm/benchmark/BenchmarkRunner.kt

class BenchmarkRunner @Inject constructor(
    private val inferenceEngine: InferenceEngine,
    private val thermalMonitor: ThermalMonitor,
    private val batteryMonitor: BatteryMonitor,
) {
    data class BenchmarkConfig(
        val promptTokenCounts: List<Int>  = listOf(64, 256, 512, 1024),
        val genTokenCounts: List<Int>     = listOf(128, 256, 512),
        val runsPerConfig: Int            = 3,
        val warmupRuns: Int               = 1,
        val testPrompts: List<String>     = defaultPrompts,
        val framework: Framework,
        val modelPath: String,
        val params: GenerationParams      = GenerationParams(),
    )

    fun runBenchmark(config: BenchmarkConfig): Flow<BenchmarkProgress> = flow {
        emit(BenchmarkProgress.Started)
        val results = mutableListOf<BenchmarkResult>()

        // Warm-up pass
        repeat(config.warmupRuns) {
            runSingleBenchmark(config.testPrompts.first(), 64, config.params)
        }

        config.promptTokenCounts.forEach { promptLen ->
            config.genTokenCounts.forEach { genLen ->
                repeat(config.runsPerConfig) { run ->
                    val result = runSingleBenchmark(
                        prompt    = config.testPrompts[run % config.testPrompts.size],
                        maxTokens = genLen,
                        params    = config.params.copy(maxTokens = genLen),
                        promptLen = promptLen,
                    )
                    results.add(result)
                    emit(BenchmarkProgress.ResultReady(result, results.size, totalRuns(config)))
                }
            }
        }

        val summary = BenchmarkSummary.from(results, config)
        emit(BenchmarkProgress.Completed(summary))
    }.flowOn(Dispatchers.IO)

    private suspend fun runSingleBenchmark(
        prompt: String, maxTokens: Int,
        params: GenerationParams, promptLen: Int = 0,
    ): BenchmarkResult {
        val startTemp   = thermalMonitor.getCpuTemperature()
        val startBattery = batteryMonitor.getCurrentNow()
        val startMem    = ProcessStats.getRSSMB()

        var firstTokenMs = -1L
        var tokenCount   = 0
        val startTime    = SystemClock.elapsedRealtime()

        inferenceEngine.generate(
            listOf(ChatMessage("user", prompt)),
            params.copy(maxTokens = maxTokens),
        ).collect { chunk ->
            if (firstTokenMs < 0) firstTokenMs = SystemClock.elapsedRealtime() - startTime
            tokenCount++
        }

        val totalMs  = SystemClock.elapsedRealtime() - startTime
        val endTemp  = thermalMonitor.getCpuTemperature()
        val peakMem  = ProcessStats.getRSSMB()

        return BenchmarkResult(
            promptTokens    = promptLen,
            generatedTokens = tokenCount,
            ttftMs          = firstTokenMs,
            totalMs         = totalMs,
            tgTps           = tokenCount / (totalMs / 1000.0),
            ppTps           = promptLen / ((firstTokenMs).coerceAtLeast(1) / 1000.0),
            peakMemMB       = peakMem - startMem,
            startTempC      = startTemp,
            endTempC        = endTemp,
            batteryDrainMa  = startBattery - batteryMonitor.getCurrentNow(),
        )
    }
}
```

### 10.3 Benchmark UI Features

- **Config panel** — select framework, model, prompt sizes, gen lengths, and run count.
- **Live progress** — animated progress bar with current TPS displayed in real time.
- **Results chart** — bar chart of TPS by configuration (Recharts-equivalent via MPAndroidChart).
- **Comparison mode** — run benchmarks for all three frameworks on the same model and overlay results.
- **Thermal timeline** — line chart showing temperature rise over the benchmark duration.
- **Export** — save results as CSV or JSON for external analysis.
- **Share** — generate a shareable summary card with device info and top-line numbers.

---

## 11. Wake Lock & Background Service

### 11.1 Architecture

```
User toggles "Keep server running"
        │
        ▼
LLMServerService (ForegroundService)
        ├── Acquires PowerManager.PARTIAL_WAKE_LOCK (CPU stays on)
        ├── Acquires WifiManager.WifiLock (Wi-Fi stays connected)
        ├── Shows persistent notification with server status
        │       ├── IP address + port
        │       ├── Requests served counter
        │       └── Quick-action buttons: Stop / Open Chat
        └── Starts / monitors LLMServer (Ktor)
```

### 11.2 Foreground Service

```kotlin
// server-service/src/main/java/com/localllm/server/LLMServerService.kt

@AndroidEntryPoint
class LLMServerService : Service() {

    @Inject lateinit var llmServer: LLMServer
    @Inject lateinit var serverConfig: ServerConfig

    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startServer()
            ACTION_STOP  -> stopSelf()
        }
        return START_STICKY
    }

    private fun startServer() {
        startForeground(NOTIF_ID, buildNotification())

        if (serverConfig.wakeLockEnabled) {
            wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "LocalLLM::ServerWakeLock")
                .apply { acquire() }
        }
        if (serverConfig.wifiLockEnabled) {
            wifiLock = (applicationContext.getSystemService(WIFI_SERVICE) as WifiManager)
                .createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "LocalLLM::WifiLock")
                .apply { acquire() }
        }

        llmServer.start()
    }

    override fun onDestroy() {
        llmServer.stop()
        wakeLock?.takeIf { it.isHeld }?.release()
        wifiLock?.takeIf { it.isHeld }?.release()
        super.onDestroy()
    }

    private fun buildNotification(): Notification {
        val channel = NotificationChannel(CHANNEL_ID, "LLM Server", NotificationManager.IMPORTANCE_LOW)
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("LocalLLM Server Running")
            .setContentText("Listening on port ${serverConfig.port} · 0 requests served")
            .setSmallIcon(R.drawable.ic_server)
            .setOngoing(true)
            .addAction(R.drawable.ic_stop, "Stop", stopPendingIntent())
            .addAction(R.drawable.ic_chat, "Open Chat", chatPendingIntent())
            .build()
    }

    companion object {
        const val ACTION_START = "com.localllm.START_SERVER"
        const val ACTION_STOP  = "com.localllm.STOP_SERVER"
        const val NOTIF_ID     = 1001
        const val CHANNEL_ID   = "llm_server"
    }
}
```

---

## 12. Dependencies & Gradle Configuration

### 12.1 Root `build.gradle.kts`

```kotlin
plugins {
    alias(libs.plugins.android.application)    apply false
    alias(libs.plugins.android.library)        apply false
    alias(libs.plugins.kotlin.android)         apply false
    alias(libs.plugins.kotlin.serialization)   apply false
    alias(libs.plugins.hilt.android)           apply false
    alias(libs.plugins.ksp)                    apply false
}
```

### 12.2 `libs.versions.toml`

```toml
[versions]
compileSdk             = "35"
minSdk                 = "28"
targetSdk              = "35"
kotlin                 = "2.1.0"
agp                    = "8.7.3"
hilt                   = "2.53.1"
ksp                    = "2.1.0-1.0.29"
compose-bom            = "2025.02.00"
lifecycle              = "2.8.7"
navigation             = "2.8.6"
room                   = "2.7.0"
datastore              = "1.1.1"
ktor                   = "3.0.3"
work                   = "2.10.0"
media3                 = "1.5.1"
okhttp                 = "4.12.0"
retrofit               = "2.11.0"
moshi                  = "1.15.1"
coil                   = "3.1.0"
mpandroidchart         = "3.1.0"
litert                 = "1.0.0"
mediapipe-tasks        = "0.10.14"
timber                 = "5.0.1"
coroutines             = "1.9.0"

[libraries]
# Compose
compose-bom                  = { group = "androidx.compose", name = "compose-bom", version.ref = "compose-bom" }
compose-ui                   = { group = "androidx.compose.ui", name = "ui" }
compose-ui-tooling           = { group = "androidx.compose.ui", name = "ui-tooling" }
compose-material3            = { group = "androidx.compose.material3", name = "material3" }
compose-material-icons       = { group = "androidx.compose.material", name = "material-icons-extended" }
compose-activity             = { group = "androidx.activity", name = "activity-compose", version = "1.10.0" }
compose-navigation           = { group = "androidx.navigation", name = "navigation-compose", version.ref = "navigation" }
compose-hilt-navigation      = { group = "androidx.hilt", name = "hilt-navigation-compose", version = "1.2.0" }

# Hilt
hilt-android                 = { group = "com.google.dagger", name = "hilt-android", version.ref = "hilt" }
hilt-compiler                = { group = "com.google.dagger", name = "hilt-android-compiler", version.ref = "hilt" }
hilt-work                    = { group = "androidx.hilt", name = "hilt-work", version = "1.2.0" }

# Room
room-runtime                 = { group = "androidx.room", name = "room-runtime", version.ref = "room" }
room-ktx                     = { group = "androidx.room", name = "room-ktx", version.ref = "room" }
room-compiler                = { group = "androidx.room", name = "room-compiler", version.ref = "room" }

# Ktor (server)
ktor-server-netty            = { group = "io.ktor", name = "ktor-server-netty", version.ref = "ktor" }
ktor-server-content-neg      = { group = "io.ktor", name = "ktor-server-content-negotiation", version.ref = "ktor" }
ktor-server-cors             = { group = "io.ktor", name = "ktor-server-cors", version.ref = "ktor" }
ktor-server-auth             = { group = "io.ktor", name = "ktor-server-auth", version.ref = "ktor" }
ktor-server-rate-limit       = { group = "io.ktor", name = "ktor-server-rate-limit", version.ref = "ktor" }
ktor-server-call-logging     = { group = "io.ktor", name = "ktor-server-call-logging", version.ref = "ktor" }
ktor-serialization-json      = { group = "io.ktor", name = "ktor-serialization-kotlinx-json", version.ref = "ktor" }

# Inference Backends
litert-api                   = { group = "com.google.ai.edge.litert", name = "litert", version.ref = "litert" }
litert-gpu                   = { group = "com.google.ai.edge.litert", name = "litert-gpu", version.ref = "litert" }
mediapipe-llm                = { group = "com.google.mediapipe", name = "tasks-genai", version.ref = "mediapipe-tasks" }

# MNN (AAR — hosted on JitPack or local maven)
mnn-android                  = { group = "com.github.alibaba", name = "MNN", version = "2.9.0" }

# Networking / Download
okhttp                       = { group = "com.squareup.okhttp3", name = "okhttp", version.ref = "okhttp" }
okhttp-logging               = { group = "com.squareup.okhttp3", name = "logging-interceptor", version.ref = "okhttp" }
retrofit                     = { group = "com.squareup.retrofit2", name = "retrofit", version.ref = "retrofit" }
retrofit-moshi               = { group = "com.squareup.retrofit2", name = "converter-moshi", version.ref = "retrofit" }
moshi-kotlin                 = { group = "com.squareup.moshi", name = "moshi-kotlin", version.ref = "moshi" }

# WorkManager
work-runtime                 = { group = "androidx.work", name = "work-runtime-ktx", version.ref = "work" }

# DataStore
datastore-prefs              = { group = "androidx.datastore", name = "datastore-preferences", version.ref = "datastore" }

# Coil (image loading)
coil-compose                 = { group = "io.coil-kt.coil3", name = "coil-compose", version.ref = "coil" }

# Charts
mpandroidchart               = { group = "com.github.PhilJay", name = "MPAndroidChart", version.ref = "mpandroidchart" }

# Coroutines
coroutines-android           = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-android", version.ref = "coroutines" }

# Markdown rendering
compose-markdown             = { group = "com.github.jeziellago", name = "compose-markdown", version = "0.5.0" }

# Logging
timber                       = { group = "com.jakewharton.timber", name = "timber", version.ref = "timber" }

# Serialization
kotlinx-serialization-json   = { group = "org.jetbrains.kotlinx", name = "kotlinx-serialization-json", version = "1.7.3" }
```

### 12.3 `app/build.gradle.kts` Key Sections

```kotlin
android {
    ndkVersion = "27.2.12479018"
    defaultConfig {
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DGGML_VULKAN=ON",
                    "-DGGML_OPENCL=ON",
                    "-DGGML_NNAPI=ON",
                    "-DCMAKE_BUILD_TYPE=Release",
                )
            }
        }
    }
    externalNativeBuild {
        cmake { path = "src/main/cpp/CMakeLists.txt" }
    }
    packaging {
        jniLibs { useLegacyPackaging = true }
    }
}
```

---

## 13. Recommended Models & Best Settings Guide

The app surfaces a **Hardware-Aware Guide** on the Home screen, generated dynamically from the detected `HardwareCapability`.

### 13.1 Model Recommendation Matrix

| Device Tier | RAM | Recommended Model | Quant | Framework | GPU Layers | Expected TPS |
|-------------|-----|-------------------|-------|-----------|-----------|--------------|
| Flagship (SD 8 Gen 3 / Dimensity 9300) | 12 GB+ | Gemma 3 12B IT | Q4_K_M | llama.cpp | 99 (all) | 18–30 |
| High-end (SD 8 Gen 2 / Dimensity 9200) | 8–12 GB | Gemma 3 4B IT | Q6_K | llama.cpp | 99 (all) | 25–40 |
| Mid-range (SD 7s Gen 3 / Dimensity 8300) | 6–8 GB | Qwen 2.5 3B IT | Q4_K_M | MNN | 32 | 20–35 |
| Entry-level (SD 695 / Dimensity 6300) | 4–6 GB | Phi-3.5 Mini | Q4_0 | LiteRT-LM | 0 (CPU) | 8–14 |
| Low-end | < 4 GB | SmolLM2 1.7B | Q4_0 | llama.cpp | 0 (CPU) | 5–10 |

### 13.2 Auto-Suggest Logic

```kotlin
// feature-home/src/main/java/com/localllm/home/ModelRecommender.kt

class ModelRecommender @Inject constructor(
    private val capability: HardwareCapability,
    private val modelRegistry: ModelRegistry,
) {
    fun getTopRecommendations(): List<ModelSuggestion> {
        return modelRegistry.all
            .filter { model ->
                model.minRamGb * 1024 <= capability.totalRamMB * 0.7  // leave 30% headroom
                    && model.supportedFrameworks.any { it.isCompatibleWith(capability) }
            }
            .sortedByDescending { it.qualityScore }
            .take(5)
            .map { model ->
                val framework = selectBestFramework(model, capability)
                val gpuLayers = if (model.recommendedGpuLayers == -1)
                    capability.recommendedGpuLayers else model.recommendedGpuLayers
                ModelSuggestion(model, framework, gpuLayers, buildRationale(model, capability))
            }
    }

    private fun buildRationale(model: ModelEntry, cap: HardwareCapability): String {
        val reasons = buildList {
            if (cap.gpuVendor == GpuVendor.QUALCOMM) add("Optimised for Adreno Vulkan")
            if (cap.cpuFeatures.contains(CpuFeature.SVE2)) add("SVE2 SIMD acceleration available")
            if (cap.totalRamMB >= 8192) add("Enough RAM for larger context windows")
        }
        return reasons.joinToString(" · ")
    }
}
```

### 13.3 Best Settings Tips (surfaced in UI)

- **Flash Attention ON** — Reduces KV cache memory by ~40% on Adreno 700-series.
- **KV Cache Q8_0** — Halves memory vs F16 with < 1% quality loss on 7B+ models.
- **ubatch-size 64–256** — Smaller values reduce latency for streaming; larger values increase prompt processing speed.
- **mlock ON** — Prevents model pages from being swapped out, improves consistency; needs sufficient free RAM.
- **Thread count = big-core count** — On Snapdragon 8 Gen 3, use 4 (4×Cortex-X4 + 4×A720), not all 12 cores; small cores add overhead.
- **RoPE YaRN** — Extend context beyond model's training length (e.g., 4K → 32K) with minor perplexity cost.

---

## 14. Security & Permissions

### 14.1 AndroidManifest Permissions

```xml
<!-- Network -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.ACCESS_WIFI_STATE" />
<uses-permission android:name="android.permission.CHANGE_WIFI_STATE" />

<!-- Wake lock (foreground server) -->
<uses-permission android:name="android.permission.WAKE_LOCK" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_SPECIAL_USE" />
<uses-permission android:name="android.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />

<!-- Storage (model files) -->
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />

<!-- Optional: receive boot completed to auto-restart server -->
<uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />
```

### 14.2 API Key Security

When API key authentication is enabled, keys are stored in Android **EncryptedSharedPreferences** (AES256-GCM) via Jetpack Security. Keys are never logged or included in crash reports.

### 14.3 Network Binding

The server defaults to binding on `0.0.0.0` (all interfaces) for LAN access. The UI warns users clearly when the server is accessible from the local network, and recommends enabling API key auth in that case.

---

## 15. Build Variants & Flavors

```kotlin
// app/build.gradle.kts
android {
    flavorDimensions += "backend"
    productFlavors {
        create("full") {
            dimension = "backend"
            // Includes all three inference backends
        }
        create("lite") {
            dimension = "backend"
            // LiteRT only — smallest APK for low-end devices
        }
        create("llamacpp") {
            dimension = "backend"
            // llama.cpp only — for power users
        }
    }
    buildTypes {
        debug {
            isDebuggable = true
            buildConfigField("Boolean", "SWAGGER_ENABLED", "true")
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            buildConfigField("Boolean", "SWAGGER_ENABLED", "false")
        }
    }
}
```

---

## Appendix A: Key Third-Party References

| Library / SDK | Source | Purpose |
|---------------|--------|---------|
| llama.cpp | github.com/ggerganov/llama.cpp | Core inference (llama.cpp backend) |
| MNN | github.com/alibaba/MNN | MNN inference backend |
| LiteRT (TFLite) | ai.google.dev/edge/litert | LiteRT-LM backend |
| MediaPipe Tasks GenAI | ai.google.dev/edge/mediapipe | LiteRT-LM Java API |
| Ktor | ktor.io | Embedded HTTP server |
| Hilt | dagger.dev/hilt | Dependency injection |
| Room | developer.android.com/jetpack/androidx/releases/room | Local database |
| WorkManager | developer.android.com/jetpack/androidx/releases/work | Background downloads |
| Jetpack Compose | developer.android.com/compose | UI toolkit |
| MPAndroidChart | github.com/PhilJay/MPAndroidChart | Benchmark charts |
| compose-markdown | github.com/jeziellago/compose-markdown | Markdown rendering in chat |

---

*Document maintained by the LocalLLM Studio team. Update this document when adding new inference backends, endpoint support, or hardware profiles.*
