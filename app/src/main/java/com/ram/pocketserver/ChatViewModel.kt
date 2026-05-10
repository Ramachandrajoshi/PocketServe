package com.ram.pocketserver

import androidx.lifecycle.ViewModel
import com.ram.pocketserver.inference.ChatMessage
import com.ram.pocketserver.inference.GenerationParams
import com.ram.pocketserver.inference.InferenceEngine
import com.ram.pocketserver.inference.ModelLoadConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ChatUiState(
    val modelPath: String = "",
    val prompt: String = "Hello",
    val response: String = "",
    val isModelLoaded: Boolean = false,
    val isGenerating: Boolean = false,
    val error: String? = null,
)

class ChatViewModel(
    private val inferenceEngine: InferenceEngine,
) : ViewModel() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null

    fun onModelPathChanged(value: String) {
        _uiState.update { it.copy(modelPath = value, error = null) }
    }

    fun onPromptChanged(value: String) {
        _uiState.update { it.copy(prompt = value, error = null) }
    }

    fun loadModel() {
        val path = uiState.value.modelPath.trim()
        if (path.isBlank()) {
            _uiState.update { it.copy(error = "Model path is required.") }
            return
        }

        scope.launch {
            _uiState.update { it.copy(error = null) }
            val result = inferenceEngine.loadModel(ModelLoadConfig(modelPath = path))
            _uiState.update { state ->
                state.copy(
                    isModelLoaded = result.isSuccess && inferenceEngine.isModelLoaded(),
                    error = result.exceptionOrNull()?.message,
                )
            }
        }
    }

    fun unloadModel() {
        stopGeneration()
        scope.launch {
            inferenceEngine.unloadModel()
            _uiState.update { it.copy(isModelLoaded = false, isGenerating = false, response = "", error = null) }
        }
    }

    fun generate() {
        val currentState = uiState.value
        if (!currentState.isModelLoaded) {
            _uiState.update { it.copy(error = "Load a model before generating.") }
            return
        }
        val prompt = currentState.prompt.trim()
        if (prompt.isBlank()) {
            _uiState.update { it.copy(error = "Prompt cannot be empty.") }
            return
        }

        generationJob?.cancel()
        generationJob = scope.launch {
            _uiState.update { it.copy(response = "", isGenerating = true, error = null) }
            val builder = StringBuilder()
            runCatching {
                inferenceEngine.generate(
                    messages = listOf(ChatMessage(role = "user", content = prompt)),
                    params = GenerationParams(),
                ).collect { chunk ->
                    builder.append(chunk.text)
                    _uiState.update { it.copy(response = builder.toString()) }
                }
            }.onFailure { throwable ->
                _uiState.update { it.copy(error = throwable.message ?: "Generation failed.") }
            }
            _uiState.update { it.copy(isGenerating = false) }
        }
    }

    fun stopGeneration() {
        generationJob?.cancel()
        generationJob = null
        _uiState.update { it.copy(isGenerating = false) }
    }

    override fun onCleared() {
        stopGeneration()
        scope.cancel()
    }
}
