
# Ollama Gradio GUI

<img width="2559" height="1376" alt="image" src="https://github.com/user-attachments/assets/60ddbb60-6196-4a96-88b9-936d21dd1eb2" />

A user-friendly web interface for interacting with Ollama large language models. Built with Gradio, this application provides a clean graphical interface for chatting with local LLMs, managing models, and leveraging advanced features like vision and image generation.

## Features

- **Model Management**: Browse available models, pull new models directly from the interface
- **Parameter Tuning**: Adjust temperature, top_p, context length, and seed for fine-tuned control
- **Vision Support**: Upload images for vision-capable models (llava, llama3.2-vision, etc.)
- **Image Generation**: Generate images using compatible models (flux, sdxl, etc.)
- **ComfyUI Integration**: Advanced image generation using customizable ComfyUI workflows
- **Tool Calling**: Ollama function calling for seamless image generation via natural language
- **Streaming Responses**: Real-time token-by-token response streaming
- **Persistent Settings**: Save and load your preferred configuration
- **System Prompts**: Customize model behavior with custom system prompts
- **Clickable Images**: Click generated images to open full-size in new tab

## Prerequisites

- [Ollama](https://ollama.com/) installed and running
- Python 3.8 or higher
- Windows, macOS, or Linux

## Installation

### 1. Clone or Navigate to the Project

```bash
cd path/to/MyOllama
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows (PowerShell)**:
```powershell
.venv\Scripts\Activate
```

**Windows (CMD)**:
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux**:
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Ollama

Make sure the Ollama service is running:

```bash
# Pull a model if needed
ollama pull llama3.2

# Or start the Ollama service
ollama serve
```

### 6. Run the Application

```bash
python app.py
```

The web interface will open automatically in your default browser at `http://localhost:7860`.

## Project Structure

```
MyOllama/
├── app.py                 # Main Gradio UI application
├── config.py              # Configuration management
├── config.json            # Saved user settings (auto-generated)
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py       # Package initialization
│   ├── ollama_utils.py   # Ollama API wrapper
│   └── comfyui_utils.py # ComfyUI API wrapper
├── ComfyApi/             # ComfyUI workflow JSON files
├── image_out/            # Generated images (auto-created)
├── run.bat               # Windows launcher script
└── ollama.ico            # Application icon
```

### File Descriptions

| File | Description |
|------|-------------|
| `app.py` | Main application file containing the Gradio interface, event handlers, and UI components |
| `config.py` | Handles loading and saving configuration settings to `config.json` |
| `utils/ollama_utils.py` | Wrapper functions for Ollama API calls including model listing, chat streaming, vision, and image generation |
| `utils/comfyui_utils.py` | ComfyUI API wrapper for image generation using custom workflows |
| `ComfyApi/` | Directory containing ComfyUI workflow JSON files for image generation |
| `requirements.txt` | Lists Python package dependencies (gradio, ollama, requests, websocket-client) |
| `config.json` | Saved user settings including model, parameters, and ComfyUI configuration |

## Architecture

The application follows a modular three-layer architecture:

### 1. UI Layer (`app.py`)
- **Gradio Components**: Dropdowns, sliders, textboxes, chatbot, and image inputs
- **Event Handlers**: Connect user interactions to business logic functions
- **State Management**: Manages chat history, model selection, and parameter values

### 2. Business Logic Layer (`utils/ollama_utils.py`)
- **Model Detection**: Automatically identifies vision and image generation models
- **API Integration**: Wraps Ollama Python SDK for clean API access
- **Streaming**: Implements generator-based streaming for real-time responses
- **Image Processing**: Handles base64 encoding/decoding for image I/O

### 3. Configuration Layer (`config.py`)
- **JSON Storage**: Persists user settings to `config.json`
- **Defaults**: Provides sensible default values for all parameters
- **Validation**: Merges saved config with defaults for robustness

### 4. ComfyUI Integration Layer (`utils/comfyui_utils.py`)
- **Workflow Management**: Loads and parses ComfyUI workflow JSON files
- **WebSocket Communication**: Real-time connection to ComfyUI server
- **Parameter Injection**: Dynamically updates prompt, seed, CFG, and steps
- **Image Processing**: Handles base64 encoding/decoding for generated images

### Data Flow

```
User Input (Message/Image)
        ↓
    Gradio UI (app.py)
        ↓
   Event Handler (respond/pull_new_model)
        ↓
   ollama_utils.py functions
        ↓
     Ollama API / ComfyUI API
        ↓
  Streaming Response / Image
        ↓
    Gradio Chatbot Update
```

## Usage Guide

### Selecting a Model

1. Use the dropdown to select an available model
2. Click "Pull" and enter a model name (e.g., `llama3.2`, `mistral`) to download new models

### Adjusting Parameters

- **Temperature**: Controls randomness (0.0 = deterministic, 2.0 = creative)
- **Top P**: Nucleus sampling threshold (lower = more focused)
- **Seed**: Set for reproducible outputs (click 🎲 for random)
- **Context Length**: Maximum tokens the model can consider

### Vision Models

When a vision model is selected (e.g., `llava`, `llama3.2-vision`), an image upload option appears. Upload an image and ask questions about it.

### Image Generation

For image generation models (e.g., `flux`, `sdxl`), the model will generate and display images based on your text prompts.

### ComfyUI Image Generation

ComfyUI integration allows you to generate images using customizable workflows:

1. **Enable ComfyUI**: Check the "Enable ComfyUI" checkbox in the sidebar
2. **Select Workflow**: Choose a workflow from the dropdown (e.g., `api_z_image_turbo.json`)
3. **Trigger Generation**: Include "generate image" in your message along with your prompt
   - Example: "generate image a beautiful sunset over mountains"
4. **Adjust Parameters**:
   - **Seed**: Control randomness for reproducible results (click 🎲 for random)
   - **CFG Scale**: Guidance strength (lower = more creative, higher = more precise)
   - **Steps**: Number of denoising steps (more = better quality, slower)

**Note**: ComfyUI must be running on `http://127.0.0.1:8188` for image generation to work.

### Saving Settings

Click "Save Settings" to persist your current model and parameter choices. These will be restored when you next run the application.

## Configuration

Default settings are defined in `config.py`:

```python
DEFAULT_CONFIG = {
    "model": "llama3.2",
    "temperature": 0.7,
    "top_p": 0.9,
    "seed": None,
    "num_ctx": 4096,
    "system_prompt": "You are a helpful assistant.",
    "comfyui_seed": 0,
    "comfyui_cfg": 3.5,
    "comfyui_steps": 35,
    "comfyui_workflow": ""
}
```

Settings are saved to `config.json` in the project directory.

## Troubleshooting

### Cannot Connect to Ollama

Ensure Ollama is running:
```bash
ollama serve
```

### Model Not Found

Pull the model first:
```bash
ollama pull llama3.2
```

### Port Already in Use

The default port is 7860. If it's in use, modify `app.py`:
```python
demo.launch(allowed_paths=["image_out"], inbrowser=True, server_port=7861)
```

### Cannot Connect to ComfyUI

Ensure ComfyUI is running with the API enabled:
1. Start ComfyUI normally (it should auto-enable the API at port 8188)
2. Verify the server is accessible at `http://127.0.0.1:8188`
3. Check that workflow files exist in the `ComfyApi/` directory

### Workflow Not Found

If you see workflow errors:
1. Ensure JSON files are in the `ComfyApi/` directory
2. Select a valid workflow from the dropdown
3. Verify the workflow contains required nodes (CLIPTextEncode, KSampler)

## License

This project is provided as-is for educational and personal use.
