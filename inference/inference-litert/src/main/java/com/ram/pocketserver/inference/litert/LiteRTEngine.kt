package com.ram.pocketserver.inference.litert

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.ram.pocketserver.hardware.HardwareCapability
import com.ram.pocketserver.inference.ChatMessage
import com.ram.pocketserver.inference.GenerationParams
import com.ram.pocketserver.inference.InferenceEngine
import com.ram.pocketserver.inference.ModelFormat
import com.ram.pocketserver.inference.ModelInfo
import com.ram.pocketserver.inference.ModelLoadConfig
import com.ram.pocketserver.inference.TokenChunk
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import javax.inject.Inject

class LiteRTEngine @Inject constructor(
    @ApplicationContext private val context: Context,
    private val hardwareCapability: HardwareCapability,
) : InferenceEngine {
    companion object {
        private const val TAG = "LiteRTEngine"
        private const val TOKEN_BUFFER_CAPACITY = 64
        // Reflection is used here to support MediaPipe versions where Backend may not be directly accessible.
        private const val LLM_BACKEND_CLASS_NAME =
            "com.google.mediapipe.tasks.genai.llminference.LlmInference\$Backend"
    }

    override val name = "LiteRT-LM"
    override val supportedFormats = listOf(ModelFormat.TFLITE_TASK, ModelFormat.SAFETENSORS)

    private var llmInference: LlmInference? = null
    private var resultChannel: Channel<TokenChunk>? = null
    private var loadedModelPath: String? = null
    private val generationMutex = Mutex()

    private fun closeAndClearResultChannel(channel: Channel<TokenChunk>, cause: Throwable? = null) {
        channel.close(cause)
        if (resultChannel === channel) {
            resultChannel = null
        }
    }

    override suspend fun loadModel(config: ModelLoadConfig): Result<Unit> {
        return withContext(Dispatchers.IO) {
            runCatching {
                val seed = runCatching { Math.toIntExact(config.seed) }.getOrElse {
                    throw IllegalArgumentException(
                        "Seed value ${config.seed} must be within Int range (${Int.MIN_VALUE} to ${Int.MAX_VALUE}) for LiteRT.",
                        it,
                    )
                }
                val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(config.modelPath)
                    .setMaxTokens(config.contextLength)
                    .setTopK(config.topK)
                    .setTemperature(config.temperature)
                    .setRandomSeed(seed)
                    .setResultListener { partialResult, done ->
                        val channel = resultChannel
                        if (channel != null) {
                            val sendResult = channel.trySend(TokenChunk(partialResult.orEmpty(), done))
                            if (sendResult.isFailure) {
                                closeAndClearResultChannel(channel, sendResult.exceptionOrNull())
                            } else if (done) {
                                closeAndClearResultChannel(channel)
                            }
                        }
                    }
                configurePreferredBackend(optionsBuilder)

                val options = optionsBuilder.build()
                llmInference = LlmInference.createFromOptions(context, options)
                loadedModelPath = config.modelPath
            }
        }
    }

    private fun configurePreferredBackend(builder: LlmInference.LlmInferenceOptions.Builder) {
        val backendName = if (hardwareCapability.hasGpuDelegate) "GPU" else "CPU"
        runCatching {
            val backendClass = Class.forName(LLM_BACKEND_CLASS_NAME)
            val backendValue = backendClass.enumConstants
                ?.firstOrNull { (it as? Enum<*>)?.name == backendName }
                ?: return@runCatching
            val method = builder.javaClass.methods.firstOrNull {
                it.name == "setPreferredBackend" && it.parameterTypes.size == 1
            } ?: return@runCatching
            method.invoke(builder, backendValue)
        }.onFailure { throwable ->
            Log.w(TAG, "Unable to configure preferred LiteRT backend: $backendName", throwable)
        }
    }

    override suspend fun unloadModel() {
        resultChannel?.close()
        resultChannel = null
        llmInference?.close()
        llmInference = null
        loadedModelPath = null
    }

    private fun List<ChatMessage>.toGemmaPrompt(): String {
        return this.joinToString("\n") { "${it.role}: ${it.content}" }
    }

    override fun generate(messages: List<ChatMessage>, params: GenerationParams): Flow<TokenChunk> {
        val inference = llmInference ?: throw IllegalStateException("Model is not loaded. Call loadModel() first.")
        return flow {
            generationMutex.withLock {
                val prompt = messages.toGemmaPrompt()
                val channel = Channel<TokenChunk>(TOKEN_BUFFER_CAPACITY)
                resultChannel = channel
                try {
                    inference.generateResponseAsync(prompt)
                    for (chunk in channel) {
                        emit(chunk)
                    }
                } finally {
                    closeAndClearResultChannel(channel)
                }
            }
        }
    }

    override suspend fun tokenize(text: String): List<Int> {
        throw UnsupportedOperationException("LiteRTEngine does not currently support tokenize().")
    }

    override suspend fun detokenize(tokens: List<Int>): String {
        throw UnsupportedOperationException("LiteRTEngine does not currently support detokenize().")
    }

    override suspend fun getModelInfo(): ModelInfo {
        val modelPath = loadedModelPath ?: throw IllegalStateException("Model is not loaded.")
        val modelName = File(modelPath).name
        return ModelInfo(modelName.ifBlank { "unnamed-litert-model" })
    }

    override fun isModelLoaded(): Boolean = llmInference != null
}
