package com.ram.pocketserver

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.ram.pocketserver.inference.InferenceEngine
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject
    lateinit var engine: InferenceEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val chatViewModel = ViewModelProvider(this, ChatViewModelFactory(engine))[ChatViewModel::class.java]
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ChatScreen(chatViewModel)
                }
            }
        }
    }
}

@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val uiState by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(text = "LiteRT Backend Test", style = MaterialTheme.typography.headlineMedium)

        OutlinedTextField(
            value = uiState.modelPath,
            onValueChange = viewModel::onModelPathChanged,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Model path") },
            singleLine = true,
        )

        OutlinedTextField(
            value = uiState.prompt,
            onValueChange = viewModel::onPromptChanged,
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp),
            label = { Text("Prompt") },
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::loadModel, enabled = !uiState.isGenerating) {
                Text("Load")
            }
            Button(onClick = viewModel::unloadModel, enabled = !uiState.isGenerating && uiState.isModelLoaded) {
                Text("Unload")
            }
            Button(onClick = viewModel::generate, enabled = !uiState.isGenerating && uiState.isModelLoaded) {
                Text("Generate")
            }
            Button(onClick = viewModel::stopGeneration, enabled = uiState.isGenerating) {
                Text("Stop")
            }
        }

        Text(
            text = if (uiState.isModelLoaded) "Model: loaded" else "Model: not loaded",
            style = MaterialTheme.typography.bodyMedium,
        )
        uiState.error?.let {
            Text(text = it, color = MaterialTheme.colorScheme.error)
        }

        Spacer(modifier = Modifier.height(4.dp))
        Text(text = "Response", style = MaterialTheme.typography.titleMedium)
        Text(text = uiState.response.ifBlank { "No response yet." })
    }
}

private class ChatViewModelFactory(
    private val engine: InferenceEngine,
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(ChatViewModel::class.java)) {
            return ChatViewModel(engine) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
    }
}
