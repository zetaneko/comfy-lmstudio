# ComfyUI LM Studio

Seamlessly integrate [LM Studio](https://lmstudio.ai/) with ComfyUI to use local LLMs (Large Language Models) directly in your workflows.

## Overview

ComfyUI LM Studio lets you hot-swap between image generation models and language models within a single workflow. Perfect for AI-assisted workflows like:

- **Caption Enhancement**: Convert tag lists into natural language descriptions before image generation
- **Prompt Engineering**: Use an LLM to rewrite, expand, or stylize your prompts
- **Vision Models**: Analyze generated images with VLMs (Vision-Language Models)
- **Creative Workflows**: Generate image descriptions, stories, or metadata alongside images

The nodes automatically manage VRAM, allowing you to unload ComfyUI models, run an LLM, then reload for image generation—all in one workflow.

## Requirements

- [ComfyUI](https://docs.comfy.org/get_started)
- [LM Studio](https://lmstudio.ai/) running locally with the server enabled
- Python 3.9+

## Installation

### Via ComfyUI Manager (Recommended)

1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for "LM Studio" in ComfyUI Manager
3. Install and restart ComfyUI

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zetaneko/comfy-lmstudio.git
cd comfy-lmstudio
pip install -e .
```

Restart ComfyUI after installation.

## Quick Start

### Simple Text Generation

1. Start LM Studio and enable the local server (default: `localhost:1234`)
2. Load a model in LM Studio
3. In ComfyUI, add an **LM Studio Quick Chat** node
4. Set your prompt and generate

### Full Workflow Setup

For more control, use the modular nodes:

```
LM Studio Connection → LM Studio Load Model → LM Studio Chat → LM Studio Unload Model
```

## Available Nodes

### 🔌 LM Studio Connection
Establishes connection to your local LM Studio server.

**Inputs:**
- `server_host`: Server address (default: `localhost:1234`)
- `auto_discover`: Auto-find LM Studio if connection fails

**Outputs:**
- `connection`: Connection handle for other nodes
- `status`: Connection status message

---

### 📦 LM Studio Load Model
Loads an LLM into memory from LM Studio.

**Inputs:**
- `connection`: From LM Studio Connection node
- `model_key`: Model to load (dropdown of available models)
- `load_mode`: Reuse existing or load new instance
- `seed`: Random seed for reproducible outputs (with randomize/increment controls)
- `clean_vram`: Unload ComfyUI models before loading (frees VRAM)
- `instance_id`: Optional identifier for multiple instances
- `ttl`: Auto-unload after idle time (seconds)

**Outputs:**
- `model`: Model handle for chat nodes
- `status`: Load status message

---

### 💬 LM Studio Chat
Multi-turn conversation node with full history management.

**Inputs:**
- `model`: From Load Model node
- `user_message`: Your message to the AI
- `image`: Optional image input for vision models (VLMs)
- `system_prompt`: Define AI behavior (only used for new conversations)
- `conversation`: Previous conversation state (connect output back here for multi-turn chat)
- `clear_conversation`: Start fresh conversation
- `temperature`: Creativity level (0.0-2.0)
- `max_tokens`: Generation limit (0 = unlimited)

**Outputs:**
- `response`: AI's text response
- `conversation`: Updated conversation state
- `full_history`: Formatted conversation history
- `model`: Model passthrough (for chaining to Unload node)

---

### 🗑️ LM Studio Unload Model
Frees VRAM by unloading the model.

**Inputs:**
- `model`: Model handle to unload

**Outputs:**
- `status`: Unload status message

**Features:**
- Automatically checks if model is loaded before attempting unload
- Verifies VRAM was actually freed
- Safe to run multiple times

---

### ⚡ LM Studio Quick Chat
All-in-one node for simple use cases. Combines connection, loading, chat, and optional unloading.

**Inputs:**
- `server_host`: LM Studio server address
- `model_key`: Model selector
- `user_message`: Your prompt
- `image`: Optional image for VLMs
- `system_prompt`: AI behavior
- `seed`: Reproducibility control
- `clean_vram`: Free VRAM before loading
- `temperature`: Creativity level
- `max_tokens`: Token limit
- `unload_after`: Auto-unload when done

**Outputs:**
- `response`: AI's response

---

## Example Workflow: Caption Enhancement

**Problem:** You have tags like `"1girl, blue_eyes, long_hair, smile"` but want a natural description for better image generation.

**Solution:**

```
[Your Tag List/Input]
    ↓
LM Studio Load Model
├─ clean_vram: ✅ True (unload SD models)
├─ model_key: "qwen2.5-7b-instruct"
└─ seed: 42
    ↓
LM Studio Chat
├─ system_prompt: "Convert anime tags into natural English descriptions."
├─ user_message: "1girl, blue_eyes, long_hair, smile, garden, flowers"
└─ Output: "A young woman with striking blue eyes and flowing long hair
            smiles warmly while standing in a vibrant garden filled with flowers."
    ↓
LM Studio Unload Model (free VRAM)
    ↓
[Load SD Model Back]
    ↓
[Use enhanced caption for CLIP Text Encode]
    ↓
[Generate Image]
```

## Features

✅ **Hot-Swap Models**: Automatically manage VRAM between ComfyUI and LM Studio models

✅ **Vision Support**: Use VLMs (Vision-Language Models) with image inputs

✅ **Multi-Turn Conversations**: Maintain chat history across multiple nodes

✅ **Reproducible Outputs**: Seed control with randomize/increment/fixed modes

✅ **VRAM Management**: Clean GPU memory before loading LLMs

✅ **Auto-Loading**: Chat node automatically loads models if not in memory

✅ **Verified Unloading**: Ensures models are actually freed from VRAM

## Development

### Install Development Dependencies

```bash
cd comfy-lmstudio
pip install -e .[dev]
pre-commit install
```

The `-e` flag creates a "live" install where code changes are automatically picked up on ComfyUI restart.

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
coverage run -m pytest
coverage report

# Run linter
ruff check .

# Run type checker
mypy .
```

### Pre-commit Hooks

Pre-commit runs `ruff` automatically on commit:

```bash
pre-commit run --all-files
```

## Troubleshooting

### "Failed to connect to LM Studio server"
- Ensure LM Studio is running and the server is started
- Check the server address matches (default: `localhost:1234`)
- Try enabling `auto_discover` on the Connection node

### "Model not currently loaded"
- The Load Model node should automatically wait for loading to complete
- Check LM Studio logs for loading errors
- Ensure you have enough VRAM for the model

### "Timeout waiting for model to load"
- Large models can take time to load (up to 5 minutes is allowed)
- Check your VRAM availability
- Try enabling `clean_vram` to free space

### Image input not working
- Ensure you're using a Vision-Language Model (VLM) like `qwen2-vl` or `llava`
- Connect a ComfyUI IMAGE type to the image input
- Regular text-only models cannot process images

## Links

- [LM Studio](https://lmstudio.ai/)
- [ComfyUI Documentation](https://docs.comfy.org/)
- [Report Issues](https://github.com/YOUR_USERNAME/comfy-lmstudio/issues)

## License

See [LICENSE](LICENSE) file for details.
