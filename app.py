# app.py
# -*- coding: utf-8 -*-
"""
KWAI Kolors WebUI

Author: Bard
Date: May 27, 2025
"""

import os
import uuid
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from pathvalidate import sanitize_filename
import gc

# --- Configuration ---
# Ensure output and model directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

MODEL_FOLDER = "models"
OUTPUT_FOLDER = "output"
MODEL_NAME = "Kwai-Kolors/Kolors"  # KWAI Kolors Model

# Determine device and dtype
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)  # Try primary GPU
        DEVICE = "cuda:0"
    except RuntimeError:
        print("CUDA device 0 not available or already in use. Trying cuda:1 if available.")
        try:
            torch.cuda.set_device(1)
            DEVICE = "cuda:1"
        except RuntimeError:
            print("CUDA device 1 not available. Falling back to CPU.")
            DEVICE = "cpu"
else:
    DEVICE = "cpu"

TORCH_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
VARIANT = "fp16" if DEVICE.startswith("cuda") else None

# --- Global Variables ---
pipeline = None
last_seed = -1

# --- Model Loading ---
def load_kolors_pipeline():
    global pipeline
    if pipeline is None:
        print(f"Loading Kolors model: {MODEL_NAME} to {DEVICE} with {TORCH_DTYPE}...")
        try:
            pipeline_args = {"cache_dir": MODEL_FOLDER, "torch_dtype": TORCH_DTYPE}
            if VARIANT:
                pipeline_args["revision"] = "fp16"
                pipeline_args["torch_dtype"] = torch.float16


            pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAME, **pipeline_args)

            if DEVICE.startswith("cuda"):
                print("Enabling model CPU offload for lower VRAM usage...")
                pipeline.enable_model_cpu_offload()  # Use CPU offload for VRAM efficiency
            else:
                pipeline.to(DEVICE)

            print("Kolors model loaded successfully.")
        except Exception as e:
            print(f"Error loading Kolors model: {e}")
            pipeline = None
            raise
    return pipeline


# --- Core Generation Logic ---
def generate_image(prompt: str, width: int, height: int, num_inference_steps: int, seed: int, guidance_scale: float, progress=gr.Progress(track_tqdm=True)):
    global last_seed, pipeline

    if pipeline is None:
        gr.Error("Model is not loaded. Please check console logs and restart the app.")
        return None, -1, "Error: Model not loaded. Check logs."

    current_seed = int(seed)
    if current_seed == -1:
        current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    last_seed = current_seed

    generator = torch.Generator(device="cpu").manual_seed(current_seed)

    status_message = "Generating..."
    try:
        print(f"Generating image with seed: {current_seed}, W: {width}, H: {height}, Steps: {num_inference_steps}, CFG: {guidance_scale}")
        pil_image = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images[0]

        safe_prompt_segment = sanitize_filename(prompt[:50] if prompt else "kolors_img")
        if not safe_prompt_segment.strip():
            safe_prompt_segment = "kolors_img"

        unique_id = str(uuid.uuid4())[:8]
        filename = f"{safe_prompt_segment}_{current_seed}_{unique_id}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)

        pil_image.save(filepath, 'PNG')
        status_message = f"Image saved as: {filepath}"
        print(status_message)

        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

        return pil_image, current_seed, status_message

    except Exception as e:
        print(f"Error during image generation: {e}")
        import traceback
        traceback.print_exc()
        status_message = f"Error: {str(e)}"
        return None, current_seed, f"Error: {str(e)}"


# --- UI Helper Functions ---
def reset_seed_value():
    return -1

def reuse_last_seed_value():
    global last_seed
    return last_seed if last_seed is not None else -1

# --- Gradio Interface ---
try:
    load_kolors_pipeline()
except Exception as e:
    print(f"Failed to load model on startup: {e}. The app might not function correctly.")

theme = gr.themes.Default().set(
    body_background_fill="#121212",
    body_text_color="#e0e0e0",
    body_text_color_subdued="#b0b0b0",
    block_background_fill="#1e1e1e",
    block_border_width="1px",
    block_border_color="#333333",
    block_label_background_fill="#2c2c2c",
    block_label_text_color="#e0e0e0",
    block_title_text_color="#ffffff",
    input_background_fill="#2c2c2c",
    input_border_color="#444444",
    button_primary_background_fill="#007bff",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#3a3a3a",
    button_secondary_text_color="#e0e0e0",
    slider_color="#007bff",
    slider_color_dark="#0056b3",
)


with gr.Blocks(theme=theme, css="""
    body { color: #e0e0e0; }
    .gradio-container { background-color: #121212; }

    .small-button {
        min-width: 0 !important; width: 3em; height: 3em;
        padding: 0.25em !important; line-height: 1; font-size: 1.2em;
        align-self: end; margin-left: 0.5em !important;
    }
    #seed_row .gr-form { display: flex; align-items: flex-end; }
    #seed_row .gr-number { flex-grow: 1; }

    .gr-group {
        border-radius: 12px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.25) !important;
        background-color: #1e1e1e !important;
        padding: 20px !important;
        border: 1px solid #333333;
    }

    h1 {
        font-size: 2.5em !important;
        color: #00aeff !important;
        text-align: center; margin-bottom: 0.5em !important;
    }
    h3 { color: #c0c0c0 !important; }
    .gr-markdown p {
        font-size: 1.1em; color: #b0b0b0;
        text-align: center; margin-bottom: 1.5em;
    }

    .gr-block-label { color: #e0e0e0 !important; }
    /* Explicitly styling input text color via CSS as a fallback */
    input[type='text'], input[type='number'], textarea { color: #e0e0e0 !important; }
    .gr-input { color: #e0e0e0 !important; background-color: #2c2c2c !important; border-color: #444444 !important; }
    .gr-slider label span { color: #e0e0e0 !important; }
    .gr-checkbox-label span { color: #e0e0e0 !important; }
    .gr-radio label span { color: #e0e0e0 !important; }
    .gr-dropdown label span { color: #e0e0e0 !important; }

    .gr-accordion summary { color: #c0c0c0 !important; }
""") as demo:

    gr.Markdown("# KWAI Kolors 🎨 Image Generation")
    gr.Markdown("Generate images using the KWAI Kolors model. Designed for ease of use and optimized for lower VRAM.")

    with gr.Row():
        with gr.Column(scale=2, min_width=400):
            with gr.Group():
                gr.Markdown("### ⚙️ Generation Settings")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g., A colorful abstract painting, vibrant colors, high detail",
                    lines=3
                )

                with gr.Row():
                    width_slider = gr.Slider(label="Width", minimum=256, maximum=2048, value=512, step=64)
                    height_slider = gr.Slider(label="Height", minimum=256, maximum=2048, value=512, step=64)

                steps_slider = gr.Slider(label="Inference Steps", minimum=4, maximum=100, value=25, step=1)
                guidance_slider = gr.Slider(label="Guidance Scale (CFG)", minimum=0.0, maximum=20.0, value=7.5, step=0.1)

                with gr.Row(elem_id="seed_row"):
                    seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, interactive=True)
                    random_seed_button = gr.Button("🎲", elem_classes="small-button")
                    reuse_seed_button = gr.Button("♻️", elem_classes="small-button")

                generate_button = gr.Button("Generate Image", variant="primary", scale=2)

        with gr.Column(scale=3, min_width=500):
            with gr.Group():
                gr.Markdown("### 🖼️ Generated Image")
                output_image = gr.Image(label="Output", type="pil", interactive=False, show_download_button=True, show_share_button=True)
                with gr.Accordion("Generation Details", open=False):
                    generated_seed_output = gr.Textbox(label="Used Seed", interactive=False)
                    status_output = gr.Textbox(label="Status / Filename", interactive=False, lines=2)

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, seed_input, guidance_slider],
        outputs=[output_image, generated_seed_output, status_output],
        api_name="generate_image"
    )

    random_seed_button.click(fn=reset_seed_value, inputs=[], outputs=seed_input)
    reuse_seed_button.click(fn=reuse_last_seed_value, inputs=[], outputs=seed_input)

# --- Launch ---
if __name__ == "__main__":
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(debug=True, share=False)
