import json
import os

CONFIG_FILE = "config.json"

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


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                merged = DEFAULT_CONFIG.copy()
                merged.update(loaded)
                return merged
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)