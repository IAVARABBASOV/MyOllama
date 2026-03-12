import gradio as gr
from gradio.components.chatbot import ChatMessage
import random
import base64
import os
from utils import ollama_utils
import config


def load_models():
    models = ollama_utils.list_models()
    return models if models else []


def get_random_seed():
    return str(random.randint(1, 999999999))


def pull_new_model(model_name, progress=gr.Progress()):
    if not model_name or not model_name.strip():
        return "Please enter a model name", gr.Dropdown(), gr.Image()
    
    model_name = model_name.strip()
    status_list = []
    
    def update_status(status):
        status_list.append(status)
    
    success, message = ollama_utils.pull_model(model_name, update_status)
    
    if success:
        updated_models = load_models()
        is_vision = ollama_utils.is_vision_model(model_name)
        return (
            f"✓ {message}",
            gr.Dropdown(choices=updated_models, value=model_name),
            gr.Image(visible=is_vision)
        )
    else:
        return f"✗ {message}", gr.Dropdown(), gr.Image()


def on_model_change(model_name):
    if not model_name:
        return gr.Image(visible=False)
    is_vision = ollama_utils.is_vision_model(model_name)
    return gr.Image(visible=is_vision)


def get_message_content(msg):
    if hasattr(msg, 'content'):
        content = msg.content
    elif isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        return str(msg)
    
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return str(content)


def respond(message, history, model, system_prompt, temperature, top_p, seed, num_ctx, image):
    if not model:
        yield history + [ChatMessage(role="assistant", content="Please select a model first.")]
        return
    
    if not message or not message.strip():
        yield history + [ChatMessage(role="assistant", content="Please enter a message.")]
        return
    
    messages = []
    for msg in history:
        if hasattr(msg, 'role') and hasattr(msg, 'content'):
            messages.append({"role": msg.role, "content": get_message_content(msg)})
        elif isinstance(msg, dict) and msg.get("role") and msg.get("content"):
            messages.append({"role": msg["role"], "content": get_message_content(msg)})
    messages.append({"role": "user", "content": message})
    
    seed_val = int(seed) if seed and str(seed).strip() else None
    
    if ollama_utils.is_image_generation_model(model):
        try:
            text_response, image_path = ollama_utils.generate_image(
                prompt=message,
                model=model,
                temperature=temperature,
                top_p=top_p,
                seed=seed_val,
                num_ctx=int(num_ctx)
            )
            new_history = history.copy() if history else []
            new_history.append(ChatMessage(role="user", content=message))
            if image_path:
                content = [
                    {"type": "text", "text": text_response},
                    {"type": "image", "url": f"file://{image_path}"}
                ]
                new_history.append(ChatMessage(role="assistant", content=content))
            else:
                new_history.append(ChatMessage(role="assistant", content=text_response))
            yield new_history
        except Exception as e:
            error_msg = str(e)
            yield history + [ChatMessage(role="assistant", content=f"Error: {error_msg}")]
        return
    
    images = None
    if image is not None:
        if not ollama_utils.is_vision_model(model):
            yield history + [ChatMessage(role="assistant", content="This model does not support image input. Please select a vision model (e.g., llava, llama3.2-vision) or remove the image.")]
            return
        try:
            with open(image, "rb") as f:
                images = [base64.b64encode(f.read()).decode('utf-8')]
        except Exception as e:
            yield history + [ChatMessage(role="assistant", content=f"Error processing image: {str(e)}")]
            return
    
    response = ""
    try:
        for chunk in ollama_utils.stream_chat(
            messages=messages,
            model=model,
            system_prompt=system_prompt.strip() if system_prompt else None,
            temperature=temperature,
            top_p=top_p,
            seed=seed_val,
            num_ctx=int(num_ctx),
            images=images
        ):
            response += chunk
            new_history = history.copy() if history else []
            new_history.append(ChatMessage(role="user", content=message))
            new_history.append(ChatMessage(role="assistant", content=response))
            yield new_history
    except Exception as e:
        error_msg = str(e)
        yield history + [ChatMessage(role="assistant", content=f"Error: {error_msg}")]


def save_settings(model, temperature, top_p, seed, num_ctx, system_prompt):
    cfg = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed if seed and str(seed).strip() else None,
        "num_ctx": int(num_ctx),
        "system_prompt": system_prompt
    }
    config.save_config(cfg)
    return "Settings saved!"


def load_settings():
    cfg = config.load_config()
    return (
        cfg.get("model", config.DEFAULT_CONFIG["model"]),
        cfg.get("temperature", config.DEFAULT_CONFIG["temperature"]),
        cfg.get("top_p", config.DEFAULT_CONFIG["top_p"]),
        str(cfg.get("seed", "")) if cfg.get("seed") else "",
        cfg.get("num_ctx", config.DEFAULT_CONFIG["num_ctx"]),
        cfg.get("system_prompt", config.DEFAULT_CONFIG["system_prompt"])
    )


def create_interface():
    cfg = config.load_config()
    models = load_models()
    initial_model = cfg.get("model", config.DEFAULT_CONFIG["model"])
    is_vision = ollama_utils.is_vision_model(initial_model) if initial_model else False
    
    with gr.Blocks(title="Ollama GUI") as demo:
        gr.Markdown("# 🦙 Ollama GUI")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection")
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=initial_model if initial_model in models else (models[0] if models else None),
                    label="Select Model",
                    interactive=True
                )
                
                with gr.Row():
                    new_model_input = gr.Textbox(label="Pull New Model", placeholder="e.g., llama3.2, mistral")
                    pull_btn = gr.Button("Pull", variant="secondary")
                
                pull_status = gr.Textbox(label="Pull Status", interactive=False)
                
                gr.Markdown("### Parameters")
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.1,
                    value=cfg.get("temperature", 0.7),
                    label="Temperature"
                )
                top_p_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05,
                    value=cfg.get("top_p", 0.9),
                    label="Top P"
                )
                
                with gr.Row():
                    seed_input = gr.Textbox(
                        value=str(cfg.get("seed", "")) if cfg.get("seed") else "",
                        label="Seed (optional)"
                    )
                    random_seed_btn = gr.Button("🎲 Random", size="sm")
                
                num_ctx_slider = gr.Slider(
                    minimum=512, maximum=32768, step=512,
                    value=cfg.get("num_ctx", 4096),
                    label="Context Length"
                )
                
                gr.Markdown("### System Prompt")
                system_prompt_box = gr.Textbox(
                    value=cfg.get("system_prompt", "You are a helpful assistant."),
                    label="System Prompt",
                    lines=3
                )
                
                with gr.Row():
                    save_btn = gr.Button("Save Settings", variant="primary")
                    load_btn = gr.Button("Load Settings")
                
                save_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=3):
                gr.Markdown("### Image Input (Vision Models Only)")
                image_input = gr.Image(
                    label="Upload Image",
                    type="filepath",
                    visible=is_vision
                )
                
                chatbot = gr.Chatbot(height=450, label="Chat")
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")
        
        model_dropdown.change(
            on_model_change,
            inputs=[model_dropdown],
            outputs=[image_input]
        )
        
        random_seed_btn.click(
            lambda: get_random_seed(),
            outputs=[seed_input]
        )
        
        pull_btn.click(
            pull_new_model,
            inputs=[new_model_input],
            outputs=[pull_status, model_dropdown, image_input]
        )
        
        submit_btn.click(
            respond,
            inputs=[msg_input, chatbot, model_dropdown, system_prompt_box,
                    temperature_slider, top_p_slider, seed_input, num_ctx_slider, image_input],
            outputs=[chatbot]
        ).then(lambda: "", outputs=[msg_input])
        
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot, model_dropdown, system_prompt_box,
                    temperature_slider, top_p_slider, seed_input, num_ctx_slider, image_input],
            outputs=[chatbot]
        ).then(lambda: "", outputs=[msg_input])
        
        clear_btn.click(
            lambda: ([], None),
            outputs=[chatbot, image_input]
        )
        
        save_btn.click(
            save_settings,
            inputs=[model_dropdown, temperature_slider, top_p_slider, seed_input,
                    num_ctx_slider, system_prompt_box],
            outputs=[save_status]
        )
        
        load_btn.click(
            lambda: load_settings(),
            outputs=[model_dropdown, temperature_slider, top_p_slider, seed_input,
                     num_ctx_slider, system_prompt_box]
        )
    
    return demo


if __name__ == "__main__":
    if not ollama_utils.check_ollama_connection():
        print("Warning: Cannot connect to Ollama. Make sure Ollama is running.")
    
    demo = create_interface()
    demo.launch(allowed_paths=["image_out"], inbrowser=True)