from inspect import cleandoc
import lmstudio as lms
import numpy as np
from PIL import Image
import tempfile
import os
import gc

# ComfyUI model management for VRAM cleanup
try:
    import comfy.model_management as mm
except ImportError:
    mm = None  # Fallback if not available


def clean_gpu_vram():
    """
    Clean GPU VRAM by unloading ComfyUI models and clearing cache.
    This makes room for LM Studio models to load.
    """
    if mm is not None:
        gc.collect()
        mm.unload_all_models()
        mm.soft_empty_cache()
    else:
        # Fallback if ComfyUI model management not available
        gc.collect()


def get_downloaded_models_list(server_host=None):
    """
    Helper function to get list of downloaded LLM models from LM Studio server.

    Args:
        server_host: Optional server host string. If None, tries to find default.

    Returns:
        list: List of model keys (strings), or ["No models available"] if none found or error
    """
    try:
        if server_host:
            with lms.Client(server_host) as client:
                models = client.llm.list_downloaded()
        else:
            # Try to find default server
            default_host = lms.Client.find_default_local_api_host()
            if default_host:
                with lms.Client(default_host) as client:
                    models = client.llm.list_downloaded()
            else:
                return ["No LM Studio server found"]

        model_keys = [model.model_key for model in models]
        return model_keys if model_keys else ["No models available"]
    except Exception as e:
        return [f"Error: {str(e)}"]


class LMStudioConnection:
    """
    Connect to an LM Studio server instance.

    This node establishes a connection to an LM Studio API server and validates
    that the server is running and accessible. The connection information can be
    passed to other LM Studio nodes for model operations.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Configure connection parameters for LM Studio server.

        Returns:
            dict: Input field configuration with server host settings.
        """
        return {
            "required": {
                "server_host": ("STRING", {
                    "default": "localhost:1234",
                    "multiline": False,
                    "tooltip": "LM Studio server address in format 'host:port' (e.g., 'localhost:1234')"
                }),
                "auto_discover": (["disabled", "enabled"], {
                    "tooltip": "Automatically discover LM Studio server on default local ports if connection fails"
                }),
            },
        }

    RETURN_TYPES = ("LMSTUDIO_CONNECTION", "STRING",)
    RETURN_NAMES = ("connection", "status",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "connect"
    CATEGORY = "LM Studio"

    def connect(self, server_host, auto_discover):
        """
        Establish connection to LM Studio server.

        Args:
            server_host: Server address in 'host:port' format
            auto_discover: Whether to auto-discover server if explicit host fails

        Returns:
            tuple: (connection_info_dict, status_message)
        """
        # First try the specified host
        if lms.Client.is_valid_api_host(server_host):
            connection_info = {
                "host": server_host,
                "discovered": False
            }
            status = f"Connected to LM Studio server at {server_host}"
            return (connection_info, status)

        # If auto-discovery is enabled, try to find server on default ports
        if auto_discover == "enabled":
            discovered_host = lms.Client.find_default_local_api_host()
            if discovered_host is not None:
                connection_info = {
                    "host": discovered_host,
                    "discovered": True
                }
                status = f"Auto-discovered LM Studio server at {discovered_host}"
                return (connection_info, status)

        # Connection failed
        error_msg = f"Failed to connect to LM Studio server at {server_host}"
        if auto_discover == "enabled":
            error_msg += " and no server found on default local ports"

        # Return None for connection but still return status message
        return (None, error_msg)


class LMStudioLoadModel:
    """
    Load an LLM model from LM Studio.

    This node loads a model into memory from your LM Studio server. You can select
    from downloaded models or manually specify a model key. The loaded model can be
    passed to inference nodes for text generation.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Configure inputs for model loading.

        Returns:
            dict: Input field configuration for model selection and loading options.
        """
        # Get list of available models for the dropdown
        # This will be called when the node is created/refreshed
        available_models = get_downloaded_models_list()

        return {
            "required": {
                "connection": ("LMSTUDIO_CONNECTION", {
                    "tooltip": "LM Studio server connection from LMStudioConnection node"
                }),
                "model_key": (available_models, {
                    "tooltip": "Select a downloaded model or refresh node to update list"
                }),
                "load_mode": (["get_or_load", "load_new_instance"], {
                    "tooltip": "get_or_load: reuse if already loaded; load_new_instance: always load new"
                }),
            },
            "optional": {
                "instance_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional instance identifier (only used with load_new_instance mode)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible outputs (0 = random). Set seed to ensure same results across runs."
                }),
                "clean_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload ComfyUI models and clear VRAM before loading LM Studio model"
                }),
                "ttl": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 86400,
                    "step": 60,
                    "tooltip": "Time to live in seconds (0 = no auto-unload). Model unloads after idle time."
                }),
            }
        }

    RETURN_TYPES = ("LMSTUDIO_MODEL", "STRING",)
    RETURN_NAMES = ("model", "status",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "load_model"
    CATEGORY = "LM Studio"

    def load_model(self, connection, model_key, load_mode, instance_id="", seed=0, clean_vram=False, ttl=0):
        """
        Load a model from LM Studio server.

        Args:
            connection: Connection info dict from LMStudioConnection
            model_key: Model identifier to load
            load_mode: Whether to get existing or load new instance
            instance_id: Optional instance identifier for new instances
            seed: Random seed for reproducible outputs (0 = random)
            clean_vram: Whether to clean VRAM before loading
            ttl: Time to live in seconds for auto-unload

        Returns:
            tuple: (model_handle_dict, status_message)
        """
        if connection is None:
            return (None, "Error: No valid connection provided")

        try:
            # Clean VRAM if requested
            if clean_vram:
                clean_gpu_vram()

            server_host = connection["host"]

            with lms.Client(server_host) as client:
                # Count loaded models before
                loaded_before = len(client.llm.list_loaded())

                # Prepare config for model loading
                config = {}
                if seed > 0:
                    config["seed"] = seed
                if ttl > 0:
                    config["ttl"] = ttl

                if load_mode == "load_new_instance":
                    # Load a new instance
                    if instance_id:
                        model = client.llm.load_new_instance(model_key, instance_id, config=config if config else None)
                        actual_instance_id = instance_id
                        status = f"Loaded new instance '{instance_id}' of model: {model_key}"
                    else:
                        model = client.llm.load_new_instance(model_key, config=config if config else None)
                        # Get the auto-generated instance ID from the model
                        actual_instance_id = getattr(model, 'instance_id', None)
                        status = f"Loaded new instance of model: {model_key}"
                else:
                    # Get existing or load if not present
                    model = client.llm.model(model_key, config=config if config else None)
                    actual_instance_id = None
                    status = f"Loaded model: {model_key}"

                # Wait for model to actually be loaded
                # The load call returns immediately but loading happens asynchronously
                import time
                max_wait_time = 300  # 5 minutes timeout
                poll_interval = 0.5  # Check every 0.5 seconds
                waited = 0

                while waited < max_wait_time:
                    loaded_after = len(client.llm.list_loaded())

                    # Check if a new model appeared in the loaded list
                    if loaded_after > loaded_before or load_mode == "get_or_load":
                        # Model is loaded, verify it's accessible
                        try:
                            # Try to access the model to confirm it's ready
                            # This will fail if the model isn't actually loaded yet
                            if actual_instance_id:
                                test_model = client.llm.model(model_key, actual_instance_id)
                            else:
                                test_model = client.llm.model(model_key)

                            # If we got here, model is ready
                            break
                        except Exception:
                            # Model not ready yet, continue waiting
                            pass

                    time.sleep(poll_interval)
                    waited += poll_interval

                if waited >= max_wait_time:
                    return (None, f"Timeout: Model took too long to load (>{max_wait_time}s)")

                # Add seed info to status if set
                if seed > 0:
                    status += f" (seed: {seed})"

                # Store only the information needed to reference the model later
                # Don't store the model object itself as it's tied to the client context
                model_handle = {
                    "host": server_host,
                    "model_key": model_key,
                    "instance_id": actual_instance_id,
                    "load_mode": load_mode,
                    "seed": seed,
                }

                return (model_handle, status)

        except Exception as e:
            error_msg = f"Failed to load model {model_key}: {str(e)}"
            return (None, error_msg)


class LMStudioUnloadModel:
    """
    Unload an LLM model from LM Studio to free VRAM.

    This node explicitly unloads a model from memory, freeing up VRAM and system
    resources. Use this when you're done with a model and want to load a different one.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Configure inputs for model unloading.

        Returns:
            dict: Input field configuration for model handle.
        """
        return {
            "required": {
                "model": ("LMSTUDIO_MODEL", {
                    "tooltip": "Model handle from LMStudioLoadModel node"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "unload_model"
    OUTPUT_NODE = True  # This is an action node that produces side effects
    CATEGORY = "LM Studio"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Force the node to always execute, even when inputs haven't changed.

        This is necessary because unloading is a side-effect operation that should
        happen every time the node is triggered, not be cached by ComfyUI.

        Returns:
            float: Current timestamp to ensure uniqueness on every execution
        """
        import time
        return time.time()

    def unload_model(self, model):
        """
        Unload a model from LM Studio server memory.

        Args:
            model: Model handle dict from LMStudioLoadModel

        Returns:
            tuple: (status_message,)
        """
        if model is None:
            return ("Error: No valid model provided",)

        try:
            server_host = model.get("host")
            model_key = model.get("model_key", "unknown")
            instance_id = model.get("instance_id")

            if not server_host:
                return ("Error: No server host information in model handle",)

            # Create client connection - don't use context manager to ensure unload completes
            client = lms.Client(server_host)

            try:
                # List all currently loaded models
                loaded_models = client.llm.list_loaded()

                # Try to find the model we want to unload
                model_found = None
                for loaded_model in loaded_models:
                    # Try to get identifying information from the loaded model
                    # Use getattr with defaults to safely access attributes
                    loaded_model_key = None

                    # Try different ways to get the model identifier
                    if hasattr(loaded_model, 'model_info'):
                        model_info = loaded_model.model_info
                        loaded_model_key = getattr(model_info, 'path', None) or getattr(model_info, 'identifier', None)

                    # Fallback: try direct attributes
                    if not loaded_model_key:
                        loaded_model_key = getattr(loaded_model, 'path', None) or getattr(loaded_model, 'identifier', None)

                    # Check if this matches our target model
                    if loaded_model_key and loaded_model_key == model_key:
                        # If instance_id is specified, check if it matches
                        if instance_id:
                            loaded_instance_id = getattr(loaded_model, 'instance_id', None)
                            if loaded_instance_id == instance_id:
                                model_found = loaded_model
                                break
                        else:
                            # No instance_id specified, take the first match
                            model_found = loaded_model
                            break

                # Check if model was found
                if model_found is None:
                    return (f"Model not currently loaded (skipping unload): {model_key}",)

                # Model is loaded, proceed with unload
                model_found.unload()

                # Wait a moment to ensure unload completes on server
                import time
                time.sleep(0.5)

                # Verify it was unloaded
                loaded_after = client.llm.list_loaded()
                still_loaded = False

                for loaded_model in loaded_after:
                    loaded_model_key = None
                    if hasattr(loaded_model, 'model_info'):
                        model_info = loaded_model.model_info
                        loaded_model_key = getattr(model_info, 'path', None) or getattr(model_info, 'identifier', None)
                    if not loaded_model_key:
                        loaded_model_key = getattr(loaded_model, 'path', None) or getattr(loaded_model, 'identifier', None)

                    if loaded_model_key and loaded_model_key == model_key:
                        if instance_id:
                            loaded_instance_id = getattr(loaded_model, 'instance_id', None)
                            if loaded_instance_id == instance_id:
                                still_loaded = True
                                break
                        else:
                            still_loaded = True
                            break

                if still_loaded:
                    status = f"Warning: Model still appears to be loaded after unload attempt: {model_key}"
                else:
                    if instance_id:
                        status = f"Successfully unloaded model instance '{instance_id}': {model_key} (freed VRAM)"
                    else:
                        status = f"Successfully unloaded model: {model_key} (freed VRAM)"

            finally:
                # Explicitly close the client
                client.close()

            return (status,)

        except Exception as e:
            error_msg = f"Failed to unload model: {str(e)}"
            return (error_msg,)


class LMStudioChat:
    """
    Have a conversation with an LLM using LM Studio.

    This node manages multi-turn conversations with an LLM. Set a system prompt to
    define the AI's behavior, send user messages, and maintain conversation history
    across multiple executions. Connect the conversation output back to the input to
    continue the chat, or enable clear_conversation to start fresh.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Configure inputs for chat conversation.

        Returns:
            dict: Input field configuration for conversation management.
        """
        return {
            "required": {
                "model": ("LMSTUDIO_MODEL", {
                    "tooltip": "Model handle from LMStudioLoadModel node"
                }),
                "user_message": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your message to the AI"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image input for vision-language models (VLMs)"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful AI assistant.",
                    "tooltip": "System prompt to set the AI's behavior (only used when starting new conversation)"
                }),
                "conversation": ("LMSTUDIO_CONVERSATION", {
                    "tooltip": "Connect this output back to continue an existing conversation"
                }),
                "clear_conversation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Start a fresh conversation, ignoring previous history"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Controls randomness (higher = more creative, lower = more focused)"
                }),
                "max_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32768,
                    "step": 1,
                    "tooltip": "Maximum tokens to generate (0 = no limit)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "LMSTUDIO_CONVERSATION", "STRING", "LMSTUDIO_MODEL")
    RETURN_NAMES = ("response", "conversation", "full_history", "model")
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "chat"
    CATEGORY = "LM Studio"

    def chat(self, model, user_message, image=None, system_prompt="You are a helpful AI assistant.",
             conversation=None, clear_conversation=False, temperature=0.7, max_tokens=0):
        """
        Generate a chat response and maintain conversation history.

        Args:
            model: Model handle dict from LMStudioLoadModel
            user_message: The user's message to send to the AI
            image: Optional image tensor from ComfyUI (for VLMs)
            system_prompt: System prompt for AI behavior (only used for new conversations)
            conversation: Previous conversation state dict (optional)
            clear_conversation: Whether to start a fresh conversation
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate (0 = no limit)

        Returns:
            tuple: (ai_response, updated_conversation_state, formatted_history, model)
        """
        if model is None:
            return ("Error: No valid model provided", None, "", None)

        if not user_message.strip():
            return ("Error: User message cannot be empty", conversation,
                    self._format_history(conversation) if conversation else "", model)

        temp_image_path = None
        try:
            server_host = model["host"]
            model_key = model["model_key"]
            instance_id = model.get("instance_id")
            seed = model.get("seed", 0)

            # Initialize or continue conversation
            if clear_conversation or conversation is None:
                # Start new conversation with system prompt
                chat = lms.Chat(system_prompt if system_prompt.strip() else None)
            else:
                # Continue existing conversation
                chat = lms.Chat.from_history(conversation)

            # Handle image input if provided
            image_handles = []
            if image is not None:
                # Convert ComfyUI image tensor to temp file
                temp_image_path = self._tensor_to_image_file(image)

                # Get model response with image
                with lms.Client(server_host) as client:
                    # Ensure model is loaded before trying to use it
                    self._ensure_model_loaded(client, model_key, instance_id, seed)

                    # Prepare image for LM Studio
                    image_handle = client.files.prepare_image(temp_image_path)
                    image_handles.append(image_handle)

                    # Add user message with image
                    chat.add_user_message(user_message, images=image_handles)

                    if instance_id:
                        llm = client.llm.model(model_key, instance_id)
                    else:
                        llm = client.llm.model(model_key)

                    # Prepare inference config
                    config = {"temperature": temperature}
                    if max_tokens > 0:
                        config["maxTokens"] = max_tokens

                    # Generate response and automatically append to chat history
                    result = llm.respond(chat, config=config, on_message=chat.append)
                    response = result.content

                    # Export conversation state
                    updated_conversation = self._export_chat_history(chat)

                    # Format history for display
                    history = self._format_history(updated_conversation)
            else:
                # No image, standard text-only chat
                chat.add_user_message(user_message)

                # Get model response
                with lms.Client(server_host) as client:
                    # Ensure model is loaded before trying to use it
                    self._ensure_model_loaded(client, model_key, instance_id, seed)

                    if instance_id:
                        llm = client.llm.model(model_key, instance_id)
                    else:
                        llm = client.llm.model(model_key)

                    # Prepare inference config
                    config = {"temperature": temperature}
                    if max_tokens > 0:
                        config["maxTokens"] = max_tokens

                    # Generate response and automatically append to chat history
                    result = llm.respond(chat, config=config, on_message=chat.append)
                    response = result.content

                    # Export conversation state
                    updated_conversation = self._export_chat_history(chat)

                    # Format history for display
                    history = self._format_history(updated_conversation)

            return (response, updated_conversation, history, model)

        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            return (error_msg, conversation,
                    self._format_history(conversation) if conversation else "", model)
        finally:
            # Clean up temp image file if it was created
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except Exception:
                    pass  # Ignore cleanup errors

    def _ensure_model_loaded(self, client, model_key, instance_id=None, seed=0):
        """
        Ensure a model is loaded before attempting to use it.

        If the model is not currently loaded, this will load it and wait for completion.

        Args:
            client: LM Studio client instance
            model_key: Model identifier
            instance_id: Optional instance ID
            seed: Random seed for model loading

        Raises:
            Exception: If model fails to load or timeout occurs
        """
        import time

        # Get the current count of loaded models
        loaded_before = len(client.llm.list_loaded())

        # Try to get/load the model
        # This will return immediately if already loaded, or start loading if not
        config = {}
        if seed > 0:
            config["seed"] = seed

        if instance_id:
            model = client.llm.model(model_key, instance_id, config=config if config else None)
        else:
            model = client.llm.model(model_key, config=config if config else None)

        # Quick check: if a new model was added, wait for it to be ready
        loaded_after = len(client.llm.list_loaded())

        # If model count didn't change, assume it was already loaded and ready
        if loaded_after == loaded_before:
            # Model was already loaded, no need to wait
            return

        # Model is newly loading, wait for it to be ready
        max_wait_time = 300  # 5 minutes
        poll_interval = 0.5
        waited = 0

        while waited < max_wait_time:
            # Check if we can successfully get a reference to the model
            try:
                if instance_id:
                    test_model = client.llm.model(model_key, instance_id)
                else:
                    test_model = client.llm.model(model_key)
                # If we got here without exception, model is ready
                return
            except Exception:
                # Model not ready yet
                pass

            time.sleep(poll_interval)
            waited += poll_interval

        raise Exception(f"Timeout waiting for model to load: {model_key}")

    def _tensor_to_image_file(self, image_tensor):
        """
        Convert ComfyUI image tensor to a temporary file.

        Args:
            image_tensor: ComfyUI IMAGE tensor (batch, height, width, channels) in range [0, 1]

        Returns:
            str: Path to temporary image file
        """
        # ComfyUI images are in format (batch, height, width, channels) with values [0, 1]
        # Take the first image from the batch
        if len(image_tensor.shape) == 4:
            image_np = image_tensor[0].cpu().numpy()
        else:
            image_np = image_tensor.cpu().numpy()

        # Convert from [0, 1] to [0, 255]
        image_np = (image_np * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='lmstudio_')
        os.close(temp_fd)  # Close the file descriptor

        # Save image to temp file
        pil_image.save(temp_path, format='PNG')

        return temp_path

    def _export_chat_history(self, chat):
        """
        Export chat object to dictionary format.

        Args:
            chat: lms.Chat object

        Returns:
            dict: Conversation state with messages list
        """
        # Try to use built-in method if available
        if hasattr(chat, 'to_history'):
            return chat.to_history()

        # Otherwise, try to access messages directly
        if hasattr(chat, 'messages'):
            return {"messages": chat.messages}

        # Fallback: return empty conversation
        return {"messages": []}

    def _format_history(self, conversation):
        """
        Format conversation history as readable text.

        Args:
            conversation: Conversation state dict

        Returns:
            str: Formatted conversation history
        """
        if not conversation or "messages" not in conversation:
            return ""

        formatted = []
        for msg in conversation["messages"]:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            if role == "SYSTEM":
                formatted.append(f"=== SYSTEM ===\n{content}")
            elif role == "USER":
                formatted.append(f">>> USER:\n{content}")
            elif role == "ASSISTANT":
                formatted.append(f"<<< ASSISTANT:\n{content}")
            else:
                formatted.append(f"[{role}]: {content}")

            formatted.append("")  # Empty line between messages

        return "\n".join(formatted)

class LMStudioQuickChat:
    """
    All-in-one LM Studio chat node for quick text generation.

    This node combines connection, model loading, chat, and optional unloading into
    a single node for simple use cases. Perfect for generating one-off text responses
    without needing to wire up multiple nodes. Supports both text and vision models.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Configure inputs for quick chat.

        Returns:
            dict: Input field configuration combining all LM Studio functionality.
        """
        available_models = get_downloaded_models_list()

        return {
            "required": {
                "server_host": ("STRING", {
                    "default": "localhost:1234",
                    "multiline": False,
                    "tooltip": "LM Studio server address (e.g., 'localhost:1234')"
                }),
                "model_key": (available_models, {
                    "tooltip": "Select a downloaded model"
                }),
                "user_message": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your message to the AI"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image input for vision-language models (VLMs)"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful AI assistant.",
                    "tooltip": "System prompt to set the AI's behavior"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible outputs (0 = random)"
                }),
                "clean_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload ComfyUI models and clear VRAM before loading LM Studio model"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Controls randomness (higher = more creative)"
                }),
                "max_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32768,
                    "step": 1,
                    "tooltip": "Maximum tokens to generate (0 = no limit)"
                }),
                "unload_after": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload model from VRAM after generation completes"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "quick_chat"
    CATEGORY = "LM Studio"

    def quick_chat(self, server_host, model_key, user_message, image=None,
                   system_prompt="You are a helpful AI assistant.", seed=0, clean_vram=False,
                   temperature=0.7, max_tokens=0, unload_after=False):
        """
        Generate a chat response with automatic connection and model management.

        Args:
            server_host: LM Studio server address
            model_key: Model identifier to use
            user_message: The user's message
            image: Optional image tensor for VLMs
            system_prompt: System prompt for AI behavior
            seed: Random seed for reproducibility
            clean_vram: Whether to clean VRAM before loading
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            unload_after: Whether to unload model after completion

        Returns:
            tuple: (ai_response,)
        """
        if not user_message.strip():
            return ("Error: User message cannot be empty",)

        temp_image_path = None
        client = None

        try:
            # Validate/connect to server
            if not lms.Client.is_valid_api_host(server_host):
                return (f"Error: Cannot connect to LM Studio server at {server_host}",)

            # Clean VRAM if requested
            if clean_vram:
                clean_gpu_vram()

            # Create client
            client = lms.Client(server_host)

            # Count loaded models before
            loaded_before = len(client.llm.list_loaded())

            # Prepare model config
            config = {}
            if seed > 0:
                config["seed"] = seed

            # Load model
            model = client.llm.model(model_key, config=config if config else None)

            # Wait for model to actually be loaded
            import time
            max_wait_time = 300  # 5 minutes timeout
            poll_interval = 0.5
            waited = 0

            while waited < max_wait_time:
                loaded_after = len(client.llm.list_loaded())

                # Check if model is ready
                if loaded_after > loaded_before or True:  # Always check since we use get_or_load mode
                    try:
                        # Try to access the model to confirm it's ready
                        test_model = client.llm.model(model_key)
                        # Model is ready
                        break
                    except Exception:
                        # Model not ready yet, continue waiting
                        pass

                time.sleep(poll_interval)
                waited += poll_interval

            if waited >= max_wait_time:
                return (f"Timeout: Model took too long to load (>{max_wait_time}s)",)

            # Create chat
            chat = lms.Chat(system_prompt if system_prompt.strip() else None)

            # Handle image if provided
            if image is not None:
                # Convert ComfyUI image tensor to temp file
                temp_image_path = self._tensor_to_image_file(image)

                # Prepare image for LM Studio
                image_handle = client.files.prepare_image(temp_image_path)

                # Add user message with image
                chat.add_user_message(user_message, images=[image_handle])
            else:
                # Add text-only user message
                chat.add_user_message(user_message)

            # Prepare inference config
            inference_config = {"temperature": temperature}
            if max_tokens > 0:
                inference_config["maxTokens"] = max_tokens

            # Generate response
            result = model.respond(chat, config=inference_config)
            response = result.content

            # Unload model if requested
            if unload_after:
                try:
                    model.unload()
                    import time
                    time.sleep(0.3)  # Brief wait for unload to complete
                except Exception as unload_error:
                    # Don't fail the whole operation if unload fails
                    response += f"\n\n(Warning: Failed to unload model: {str(unload_error)})"

            return (response,)

        except Exception as e:
            error_msg = f"Quick chat error: {str(e)}"
            return (error_msg,)
        finally:
            # Clean up temp image file if it was created
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except Exception:
                    pass  # Ignore cleanup errors

            # Close client connection
            if client:
                try:
                    client.close()
                except Exception:
                    pass  # Ignore cleanup errors

    def _tensor_to_image_file(self, image_tensor):
        """
        Convert ComfyUI image tensor to a temporary file.

        Args:
            image_tensor: ComfyUI IMAGE tensor (batch, height, width, channels) in range [0, 1]

        Returns:
            str: Path to temporary image file
        """
        # ComfyUI images are in format (batch, height, width, channels) with values [0, 1]
        # Take the first image from the batch
        if len(image_tensor.shape) == 4:
            image_np = image_tensor[0].cpu().numpy()
        else:
            image_np = image_tensor.cpu().numpy()

        # Convert from [0, 1] to [0, 255]
        image_np = (image_np * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='lmstudio_')
        os.close(temp_fd)  # Close the file descriptor

        # Save image to temp file
        pil_image.save(temp_path, format='PNG')

        return temp_path


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LMStudioConnection": LMStudioConnection,
    "LMStudioLoadModel": LMStudioLoadModel,
    "LMStudioUnloadModel": LMStudioUnloadModel,
    "LMStudioChat": LMStudioChat,
    "LMStudioQuickChat": LMStudioQuickChat
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LMStudioConnection": "LM Studio Connection",
    "LMStudioLoadModel": "LM Studio Load Model",
    "LMStudioUnloadModel": "LM Studio Unload Model",
    "LMStudioChat": "LM Studio Chat",
    "LMStudioQuickChat": "LM Studio Quick Chat"
}
