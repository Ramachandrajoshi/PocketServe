package com.ram.pocketserver.inference

import kotlinx.coroutines.flow.Flow

enum class ModelFormat { GGUF, MNN, TFLITE_TASK, SAFETENSORS }
enum class Framework { MNN, LLAMACPP, LITERT }

data class ModelLoadConfig(
    val modelPath: String,
    val contextLength: Int = 4096,
    val topK: Int = 40,
    val temperature: Float = 0.7f,
    val seed: Int = -1,
)

data class ChatMessage(val role: String, val content: String)

data class GenerationParams(
    val maxTokens: Int = 512,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val repeatPenalty: Float = 1.1f,
    val seed: Int = -1,
    val stop: List<String> = emptyList(),
    val stream: Boolean = true,
    val minP: Float = 0.05f,
    val tfsZ: Float = 1.0f,
    val typicalP: Float = 1.0f,
    val mirostatMode: Int = 0,
    val mirostatTau: Float = 5.0f,
    val mirostatEta: Float = 0.1f,
    val grammar: String? = null,
    val jsonSchema: String? = null,
)

data class TokenChunk(val text: String, val done: Boolean)

data class ModelInfo(val id: String)

interface InferenceEngine {
    val name: String
    val supportedFormats: List<ModelFormat>

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
