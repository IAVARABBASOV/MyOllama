import os
import json
import requests
import time
import base64
import threading
import websocket
from typing import Dict, List, Tuple, Optional, Any

COMFYUI_DIR = "ComfyApi"
COMFYUI_HOST = "http://127.0.0.1:8188"
COMFYUI_WS = "ws://127.0.0.1:8188/ws"
COMFYUI_URL = COMFYUI_HOST
IMAGE_OUT_DIR = "image_out"


def get_workflow_files() -> List[str]:
    if not os.path.exists(COMFYUI_DIR):
        return []
    return [f for f in os.listdir(COMFYUI_DIR) if f.endswith('.json')]


def load_workflow(workflow_file: str) -> Optional[Dict]:
    filepath = os.path.join(COMFYUI_DIR, workflow_file)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading workflow {workflow_file}: {e}")
        return None


def find_prompt_nodes(workflow: Dict) -> List[Tuple[str, Dict]]:
    prompt_nodes = []
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if "TextEncode" in class_type or class_type == "CLIPTextEncode":
            inputs = node.get("inputs", {})
            if "text" in inputs:
                prompt_nodes.append((node_id, node, "positive"))
        elif class_type == "KSampler":
            inputs = node.get("inputs", {})
            if "seed" in inputs:
                prompt_nodes.append((node_id, node, "sampler"))
    return prompt_nodes


def find_negative_prompt_nodes(workflow: Dict) -> List[Tuple[str, Dict]]:
    negative_nodes = []
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if "TextEncode" in class_type or class_type == "CLIPTextEncode":
            inputs = node.get("inputs", {})
            if "text" in inputs:
                ref = inputs.get("clip", [])
                if isinstance(ref, list) and len(ref) > 0:
                    for other_id, other_node in workflow.items():
                        if other_id.startswith("76:") and other_node.get("class_type") == "CLIPTextEncode":
                            other_inputs = other_node.get("inputs", {})
                            if "text" in other_inputs and other_inputs.get("text") == "":
                                negative_nodes.append((node_id, node))
                                break
    return negative_nodes


def get_configurable_params(workflow: Dict) -> Dict[str, Any]:
    params = {
        "positive_prompt": {"type": "text", "default": "", "node_id": None},
        "negative_prompt": {"type": "text", "default": "", "node_id": None},
        "seed": {"type": "number", "default": 0, "node_id": None},
        "cfg": {"type": "number", "default": 3.5, "node_id": None},
        "steps": {"type": "number", "default": 35, "node_id": None},
    }
    
    for node_id, node, node_type in find_prompt_nodes(workflow):
        if node_type == "positive" and params["positive_prompt"]["node_id"] is None:
            params["positive_prompt"]["node_id"] = node_id
            if "text" in node.get("inputs", {}):
                params["positive_prompt"]["default"] = node["inputs"]["text"]
    
    for node_id, node in find_negative_prompt_nodes(workflow):
        if params["negative_prompt"]["node_id"] is None:
            params["negative_prompt"]["node_id"] = node_id
    
    for node_id, node in workflow.items():
        if node.get("class_type") == "KSampler":
            inputs = node.get("inputs", {})
            if params["seed"]["node_id"] is None:
                params["seed"]["node_id"] = node_id
                params["seed"]["default"] = inputs.get("seed", 0)
            if params["cfg"]["node_id"] is None:
                params["cfg"]["node_id"] = node_id
                params["cfg"]["default"] = inputs.get("cfg", 3.5)
            if params["steps"]["node_id"] is None:
                params["steps"]["node_id"] = node_id
                params["steps"]["default"] = inputs.get("steps", 35)
    
    return params


def check_comfyui_connection() -> bool:
    try:
        response = requests.get(f"{COMFYUI_HOST}/system_stats", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def queue_prompt(workflow: Dict) -> Optional[str]:
    try:
        response = requests.post(
            f"{COMFYUI_HOST}/prompt",
            json={"prompt": workflow},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("prompt_id")
        return None
    except Exception as e:
        print(f"Error queueing prompt: {e}")
        return None


def get_prompt_status(prompt_id: str) -> Optional[Dict]:
    try:
        response = requests.get(f"{COMFYUI_HOST}/history/{prompt_id}", timeout=10)
        if response.status_code == 200:
            return response.json().get(prompt_id, {})
        return None
    except Exception:
        return None


def get_output_images(prompt_id: str) -> List[Dict]:
    try:
        response = requests.get(f"{COMFYUI_HOST}/history/{prompt_id}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            prompt_data = data.get(prompt_id, {})
            outputs = prompt_data.get("outputs", {})
            print(f"[ComfyUI] History response for {prompt_id}: {json.dumps(data, indent=2)[:500]}...")
            images = []
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        images.append({
                            "node_id": node_id,
                            "filename": img.get("filename"),
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output")
                        })
            print(f"[ComfyUI] Found {len(images)} images in output")
            return images
        return []
    except Exception as e:
        print(f"[ComfyUI] Error getting output images: {e}")
        return []


def get_image_url(subfolder: str, filename: str, image_type: str = "output") -> str:
    """Get the ComfyUI view URL for an image."""
    return f"{COMFYUI_HOST}/view?filename={filename}&subfolder={subfolder}&type={image_type}"


class ComfyUIWebSocket:
    def __init__(self, prompt_id: str, timeout: int = 300):
        self.prompt_id = prompt_id
        self.timeout = timeout
        self.completed = False
        self.error = None
        self.result = None
        self._lock = threading.Lock()
    
    def wait_for_completion(self) -> bool:
        ws = websocket.WebSocketApp(
            COMFYUI_WS,
            on_message=self._on_message,
            on_error=self._on_error,
            on_open=self._on_open,
            on_close=self._on_close
        )
        
        self.ws = ws
        
        ws_thread = threading.Thread(target=lambda: ws.run_forever())
        ws_thread.daemon = True
        ws_thread.start()
        
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            with self._lock:
                if self.completed:
                    ws.close()
                    return True
                if self.error:
                    ws.close()
                    return False
            
            try:
                status = get_prompt_status(self.prompt_id)
                if status:
                    status_str = status.get("status", {})
                    if status_str.get("completed", False):
                        print(f"[ComfyUI] Prompt {self.prompt_id} completed (detected via history)")
                        with self._lock:
                            self.completed = True
                        ws.close()
                        return True
            except:
                pass
            
            time.sleep(0.5)
        
        try:
            status = get_prompt_status(self.prompt_id)
            if status:
                status_str = status.get("status", {})
                if status_str.get("completed", False):
                    print(f"[ComfyUI] Prompt {self.prompt_id} completed (detected via history after timeout)")
                    return True
        except:
            pass
        
        ws.close()
        print(f"[ComfyUI] Timeout waiting for prompt {self.prompt_id}")
        return False
    
    def _on_open(self, ws):
        print(f"[ComfyUI] WebSocket connected")
    
    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[ComfyUI] WebSocket closed: {close_status_code} - {close_msg}")
    
    def _on_error(self, ws, error):
        print(f"[ComfyUI] WebSocket error: {error}")
        with self._lock:
            self.error = str(error)
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "status":
                status_data = data.get("data", {})
                sid = status_data.get("sid", "")
                if status_data.get("exec_info"):
                    print(f"[ComfyUI] Status: queue_remaining={status_data['exec_info'].get('queue_remaining', 'N/A')}")
            
            elif msg_type == "executed":
                node_data = data.get("data", {})
                if node_data.get("prompt_id") == self.prompt_id:
                    print(f"[ComfyUI] Prompt {self.prompt_id} executed")
                    if node_data.get("output"):
                        self.result = node_data["output"]
                    with self._lock:
                        self.completed = True
            
            elif msg_type == "executing":
                exec_data = data.get("data", {})
                if exec_data.get("prompt_id") == self.prompt_id:
                    node = exec_data.get("node", "")
                    if node:
                        print(f"[ComfyUI] Executing node: {node}")
                    else:
                        print(f"[ComfyUI] Prompt {self.prompt_id} execution started")
            
            elif msg_type == "progress":
                progress_data = data.get("data", {})
                if progress_data.get("prompt_id") == self.prompt_id:
                    print(f"[ComfyUI] Progress: {progress_data.get('value', 0)}/{progress_data.get('max', 0)}")
                    
        except Exception as e:
            print(f"[ComfyUI] Error parsing WebSocket message: {e}")


def download_image(subfolder: str, filename: str, image_type: str = "output") -> Optional[str]:
    try:
        url = f"{COMFYUI_URL}/view"
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type
        }
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            if not os.path.exists(IMAGE_OUT_DIR):
                os.makedirs(IMAGE_OUT_DIR)
            
            timestamp = int(time.time())
            save_filename = f"comfyui_{timestamp}_{filename}"
            filepath = os.path.join(IMAGE_OUT_DIR, save_filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
        return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def execute_workflow(
    workflow_file: str,
    positive_prompt: str = "",
    negative_prompt: str = "",
    seed: int = 0,
    cfg: float = 3.5,
    steps: int = 35,
    timeout: int = 300
) -> Tuple[bool, str, Optional[str]]:
    if not check_comfyui_connection():
        return False, "Cannot connect to ComfyUI. Make sure ComfyUI is running.", None
    
    workflow = load_workflow(workflow_file)
    if not workflow:
        return False, f"Failed to load workflow: {workflow_file}", None
    
    params = get_configurable_params(workflow)
    
    modified_workflow = {}
    for node_id, node in workflow.items():
        modified_node = node.copy()
        inputs = modified_node.get("inputs", {}).copy()
        
        if params["positive_prompt"]["node_id"] == node_id and "text" in inputs:
            inputs["text"] = positive_prompt
        
        if params["negative_prompt"]["node_id"] == node_id and "text" in inputs:
            inputs["text"] = negative_prompt
        
        if params["seed"]["node_id"] == node_id and "seed" in inputs:
            inputs["seed"] = seed
        
        if params["cfg"]["node_id"] == node_id and "cfg" in inputs:
            inputs["cfg"] = cfg
        
        if params["steps"]["node_id"] == node_id and "steps" in inputs:
            inputs["steps"] = steps
        
        modified_node["inputs"] = inputs
        modified_workflow[node_id] = modified_node
    
    prompt_id = queue_prompt(modified_workflow)
    if not prompt_id:
        return False, "Failed to queue prompt in ComfyUI", None
    
    print(f"[ComfyUI] Prompt queued with ID: {prompt_id}")
    
    comfy_ws = ComfyUIWebSocket(prompt_id, timeout=timeout)
    success = comfy_ws.wait_for_completion()
    
    if not success:
        status = get_prompt_status(prompt_id)
        if status:
            print(f"[ComfyUI] Final status: {json.dumps(status, indent=2)}")
    
    images = get_output_images(prompt_id)
    if images:
        first_image = images[0]
        print(f"[ComfyUI] Image found: {first_image['filename']}")
        
        image_url = get_image_url(
            first_image["subfolder"],
            first_image["filename"],
            first_image["type"]
        )
        print(f"[ComfyUI] Image URL: {image_url}")
        
        local_path = download_image(
            first_image["subfolder"],
            first_image["filename"],
            first_image["type"]
        )
        if local_path:
            print(f"[ComfyUI] Image saved locally: {local_path}")
            print(f"[ComfyUI] Image URL (for reference): {image_url}")
            return True, "Image generated successfully", local_path
        else:
            print(f"[ComfyUI] Failed to save image locally, using URL: {image_url}")
            return True, "Image generated successfully", image_url
    
    return False, "No images generated", None
