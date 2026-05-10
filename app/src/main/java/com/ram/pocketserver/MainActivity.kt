package com.ram.pocketserver

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.ram.pocketserver.inference.litert.LiteRTEngine
import com.ram.pocketserver.inference.ModelLoadConfig
import com.ram.pocketserver.inference.ChatMessage
import com.ram.pocketserver.inference.GenerationParams
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject
    lateinit var engine: LiteRTEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ChatScreen(engine)
                }
            }
        }
    }
}

@Composable
fun ChatScreen(engine: LiteRTEngine) {
    var responseText by remember { mutableStateOf("Ready") }
    val scope = rememberCoroutineScope()

    Column(modifier = Modifier.padding(16.dp)) {
        Text(text = "LiteRT Backend Test", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = {
            scope.launch {
                val config = ModelLoadConfig(modelPath = "/path/to/model.tflite")
                engine.loadModel(config)
                responseText = "Model load attempted."
            }
        }) {
            Text("Load Model (Stub)")
        }
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = {
            scope.launch {
                responseText = ""
                engine.generate(
                    listOf(ChatMessage("user", "Hello")),
                    GenerationParams()
                ).collect { chunk ->
                    responseText += chunk.text
                }
            }
        }) {
            Text("Generate")
        }
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = responseText)
    }
}
