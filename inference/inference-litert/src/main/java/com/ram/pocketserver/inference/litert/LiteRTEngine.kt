package com.ram.pocketserver.inference.litert

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
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
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import javax.inject.Inject

class LiteRTEngine @Inject constructor(
    @ApplicationContext private val context: Context,
) : InferenceEngine {
    override val name = "LiteRT-LM"
    override val supportedFormats = listOf(ModelFormat.TFLITE_TASK, ModelFormat.SAFETENSORS)

    private var llmInference: LlmInference? = null
    private var resultChannel: Channel<TokenChunk>? = null
    private var loadedModelPath: String? = null
    private val generationMutex = Mutex()

    override suspend fun loadModel(config: ModelLoadConfig): Result<Unit> {
        return withContext(Dispatchers.IO) {
            runCatching {
                require(config.seed in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) {
                    "Seed must be within Int range for LiteRT."
                }
                val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(config.modelPath)
                    .setMaxTokens(config.contextLength)
                    .setTopK(config.topK)
                    .setTemperature(config.temperature)
                    .setRandomSeed(config.seed.toInt())
                    .setResultListener { partialResult, done ->
                        val channel = resultChannel
                        if (channel != null) {
                            val sendResult = channel.trySend(TokenChunk(partialResult.orEmpty(), done))
                            if (sendResult.isFailure) {
                                channel.close(sendResult.exceptionOrNull())
                                if (resultChannel === channel) {
                                    resultChannel = null
                                }
                            } else if (done) {
                                channel.close()
                                if (resultChannel === channel) {
                                    resultChannel = null
                                }
                            }
                        }
                    }

                val options = optionsBuilder.build()
                llmInference = LlmInference.createFromOptions(context, options)
                loadedModelPath = config.modelPath
            }
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
                val channel = Channel<TokenChunk>(Channel.BUFFERED)
                resultChannel = channel
                try {
                    inference.generateResponseAsync(prompt)
                    for (chunk in channel) {
                        emit(chunk)
                    }
                } finally {
                    channel.close()
                    if (resultChannel === channel) {
                        resultChannel = null
                    }
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
        return ModelInfo(File(modelPath).name.ifBlank { modelPath })
    }

    override fun isModelLoaded(): Boolean = llmInference != null
}
