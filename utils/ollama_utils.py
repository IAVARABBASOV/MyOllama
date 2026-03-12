import ollama
import base64
import os


VISION_MODELS = [
    "llava", "llava-llama3", "llava-v1.5", "llava-v1.6",
    "llama3.2-vision", "llama3.2:11b-vision", "llama3.2:90b-vision",
    "bakllava", "moondream", "minicpm-v", "cogvlm",
    "llava:7b", "llava:13b", "llava:34b",
    "x/llava", "x/llava-llama3",
    "qwen2.5-vl", "qwen2.5:vl", "qwen2.5vl",
    "qwen3.5-vl", "qwen3.5:vl", "qwen3.5vl",
    "qwen-vl", "qwen:vl",
    "vision",
]

IMAGE_GEN_MODELS = [
    "flux", "flux1", "flux.1",
    "z-image-turbo", "z-image", "zimage-turbo",
    "sdxl", "stable-diffusion",
    "dall-e", "dalle",
    "imagegen", "image-gen",
    " Playground", "playground",
]


def is_vision_model(model_name):
    model_lower = model_name.lower()
    for vision_model in VISION_MODELS:
        if vision_model in model_lower or model_lower.startswith(vision_model.split(":")[0]):
            return True
    try:
        model_info = ollama.show(model_name)
        model_str = str(model_info).lower()
        if 'vision' in model_str or 'multimodal' in model_str:
            return True
        if hasattr(model_info, 'parameters') and model_info.parameters:
            if 'vision' in str(model_info.parameters).lower() or 'multimodal' in str(model_info.parameters).lower():
                return True
    except Exception:
        pass
    return False


def is_image_generation_model(model_name):
    model_lower = model_name.lower()
    for img_model in IMAGE_GEN_MODELS:
        if img_model in model_lower or model_lower.startswith(img_model.split(":")[0]):
            return True
    return False


def list_models():
    try:
        models = ollama.list()
        if hasattr(models, 'models'):
            return [m.model for m in models.models]
        return []
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def pull_model(model_name, callback=None):
    try:
        for progress in ollama.pull(model_name, stream=True):
            if callback and 'status' in progress:
                callback(progress['status'])
        return True, f"Successfully pulled {model_name}"
    except Exception as e:
        return False, str(e)


def stream_chat(messages, model, system_prompt=None, temperature=0.7, top_p=0.9, seed=None, num_ctx=4096, images=None):
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_ctx": num_ctx,
    }
    if seed is not None:
        options["seed"] = seed
    
    formatted_messages = []
    
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    
    for msg in messages:
        formatted_msg = {"role": msg["role"], "content": msg["content"]}
        if images and msg["role"] == "user":
            formatted_msg["images"] = images
        formatted_messages.append(formatted_msg)
    
    try:
        stream = ollama.chat(
            model=model,
            messages=formatted_messages,
            stream=True,
            options=options
        )
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    except Exception as e:
        yield f"Error: {str(e)}"


IMAGE_OUT_DIR = "image_out"


def generate_image(prompt, model, temperature=0.7, top_p=0.9, seed=None, num_ctx=4096):
    if not os.path.exists(IMAGE_OUT_DIR):
        os.makedirs(IMAGE_OUT_DIR)
    
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_ctx": num_ctx,
    }
    if seed is not None:
        options["seed"] = seed
    
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options=options
        )
        
        text_response = response.get("response", "")
        images_data = response.get("images", [])
        
        saved_image_path = None
        if images_data:
            image_base64 = images_data[0]
            image_data = base64.b64decode(image_base64)
            
            import time
            timestamp = int(time.time())
            filename = f"generated_{timestamp}.png"
            filepath = os.path.join(IMAGE_OUT_DIR, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_data)
            saved_image_path = filepath
        
        return text_response, saved_image_path
        
    except Exception as e:
        return f"Error: {str(e)}", None


def check_ollama_connection():
    try:
        ollama.list()
        return True
    except Exception:
        return False