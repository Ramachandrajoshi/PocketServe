package com.localllm.inference.litert

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.localllm.hardware.HardwareCapability
import com.localllm.inference.ChatMessage
import com.localllm.inference.GenerationParams
import com.localllm.inference.InferenceEngine
import com.localllm.inference.ModelFormat
import com.localllm.inference.ModelInfo
import com.localllm.inference.ModelLoadConfig
import com.localllm.inference.TokenChunk
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import javax.inject.Inject

class LiteRTEngine @Inject constructor(
    @ApplicationContext private val context: Context,
    private val hardwareCapability: HardwareCapability,
) : InferenceEngine {
    override val name = "LiteRT-LM"
    override val supportedFormats = listOf(ModelFormat.TFLITE_TASK, ModelFormat.SAFETENSORS)

    private var llmInference: LlmInference? = null
    private var resultChannel: Channel<TokenChunk>? = null

    override suspend fun loadModel(config: ModelLoadConfig): Result<Unit> {
        return withContext(Dispatchers.IO) {
            runCatching {
                val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(config.modelPath)
                    .setMaxTokens(config.contextLength)
                    .setTopK(config.topK)
                    .setTemperature(config.temperature)
                    .setRandomSeed(config.seed.toInt())
                    .setResultListener { partialResult, done ->
                        resultChannel?.trySend(TokenChunk(partialResult ?: "", done))
                        if (done) {
                            resultChannel?.close()
                        }
                    }

                val options = optionsBuilder.build()
                llmInference = LlmInference.createFromOptions(context, options)
            }
        }
    }

    override suspend fun unloadModel() {
        llmInference?.close()
        llmInference = null
    }

    private fun List<ChatMessage>.toGemmaPrompt(): String {
        return this.joinToString("\n") { "${it.role}: ${it.content}" }
    }

    override fun generate(messages: List<ChatMessage>, params: GenerationParams): Flow<TokenChunk> {
        val prompt = messages.toGemmaPrompt()
        val channel = Channel<TokenChunk>(Channel.UNLIMITED)
        resultChannel = channel

        // Start generation asynchronously. The listener provided during loadModel will push to the channel.
        llmInference?.generateResponseAsync(prompt)

        return channel.receiveAsFlow()
    }

    override suspend fun tokenize(text: String): List<Int> = emptyList()
    override suspend fun detokenize(tokens: List<Int>): String = ""

    override suspend fun getModelInfo(): ModelInfo = ModelInfo("litert-model")

    override fun isModelLoaded(): Boolean = llmInference != null
}
