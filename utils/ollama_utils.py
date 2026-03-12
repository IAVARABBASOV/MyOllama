import ollama
import base64
import os
import json
from typing import Dict, List, Any, Optional, Callable


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


TOOL_DEFINITIONS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image using ComfyUI. Use when user wants to create, draw, visualize, or generate any image. The model should call this tool automatically when the user asks for an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed description of the image to generate. Include subject, style, lighting, mood, and specific details."
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducibility. Use 0 for random.",
                            "default": 0
                        },
                        "cfg": {
                            "type": "number",
                            "description": "CFG scale (guidance scale). Higher values = more adherence to prompt. Recommended: 3.0-7.0",
                            "default": 3.5
                        },
                        "steps": {
                            "type": "integer",
                            "description": "Number of sampling steps. More steps = more detail but slower. Recommended: 20-50",
                            "default": 35
                        },
                        "workflow": {
                            "type": "string",
                            "description": "The ComfyUI workflow file to use (e.g., 'api_z_image_turbo.json'). If not specified, uses the default workflow.",
                            "default": ""
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    ]
}


def get_tool_definitions() -> Dict:
    from utils import comfyui_utils
    workflows = comfyui_utils.get_workflow_files()
    
    tool_defs = json.loads(json.dumps(TOOL_DEFINITIONS))
    
    if workflows:
        workflow_list = ", ".join(workflows)
        tool_defs["tools"][0]["function"]["parameters"]["properties"]["workflow"]["description"] = f"ComfyUI workflow file from: {workflow_list}"
        tool_defs["tools"][0]["function"]["parameters"]["properties"]["workflow"]["enum"] = workflows
    
    return tool_defs


def execute_tool_call(tool_name: str, arguments: Dict, default_workflow: str = "", 
                       comfyui_seed: int = 0, comfyui_cfg: float = 3.5, comfyui_steps: int = 35) -> Dict[str, Any]:
    from utils import comfyui_utils
    
    if tool_name == "generate_image":
        prompt = arguments.get("prompt", "")
        seed = comfyui_seed
        cfg = comfyui_cfg
        steps = comfyui_steps
        workflow = arguments.get("workflow", default_workflow)
        
        if not workflow:
            workflows = comfyui_utils.get_workflow_files()
            workflow = workflows[0] if workflows else ""
        
        if not workflow:
            return {
                "success": False,
                "message": "No ComfyUI workflow available. Please add a workflow JSON file to the ComfyApi folder.",
                "image_path": None
            }
        
        success, message, image_path = comfyui_utils.execute_workflow(
            workflow_file=workflow,
            positive_prompt=prompt,
            negative_prompt="",
            seed=seed,
            cfg=cfg,
            steps=steps
        )
        
        return {
            "success": success,
            "message": message,
            "image_path": image_path,
            "prompt": prompt
        }
    
    return {
        "success": False,
        "message": f"Unknown tool: {tool_name}",
        "image_path": None
    }


def stream_chat_with_tools(
    messages: List[Dict],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    num_ctx: int = 4096,
    images: Optional[List[str]] = None,
    default_workflow: str = "",
    comfyui_seed: int = 0,
    comfyui_cfg: float = 3.5,
    comfyui_steps: int = 35,
    tool_callback: Optional[Callable] = None
):
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
    
    tool_defs = get_tool_definitions()
    
    try:
        stream = ollama.chat(
            model=model,
            messages=formatted_messages,
            stream=True,
            options=options,
            tools=tool_defs["tools"]
        )
        
        for chunk in stream:
            if 'message' in chunk:
                msg = chunk['message']
                
                if 'tool_calls' in msg and msg['tool_calls']:
                    for tool_call in msg['tool_calls']:
                        func = tool_call.get('function', {})
                        tool_name = func.get('name')
                        arguments = func.get('arguments', {})
                        
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except:
                                arguments = {}
                        
                        if tool_name == "generate_image":
                            result = execute_tool_call(tool_name, arguments, default_workflow, 
                                                       comfyui_seed, comfyui_cfg, comfyui_steps)
                            
                            if tool_callback:
                                tool_callback(tool_name, arguments, result)
                            
                            if result["success"] and result["image_path"]:
                                yield {
                                    "type": "tool_result",
                                    "tool": tool_name,
                                    "success": True,
                                    "message": result["message"],
                                    "image_path": result["image_path"],
                                    "prompt": result.get("prompt", "")
                                }
                            else:
                                yield {
                                    "type": "tool_result",
                                    "tool": tool_name,
                                    "success": False,
                                    "message": result.get("message", result.get("error", "Unknown error")),
                                    "image_path": None
                                }
                
                if 'content' in msg and msg['content']:
                    yield {
                        "type": "content",
                        "content": msg['content']
                    }
                    
    except Exception as e:
        yield {"type": "error", "message": str(e)}