# KolorForge-Kolors-WebUI

This is a simple web UI for generating images using the [KWAI Kolors](https://huggingface.co/Kwai-Kolors/Kolors) model. It's designed to be easy to use and optimized for lower VRAM usage.

## Features

*   Clean and intuitive Gradio interface.
*   Dark mode theme.
*   Optimized for lower VRAM usage with CPU offloading.
*   Saves generated images to the `output` folder.
*   Seed control for reproducible results.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Run the `install.bat` script:**

    *   This script will create a Python virtual environment, activate it, and install the necessary dependencies, including PyTorch.

3.  **If PyTorch installation fails:**

    *   Visit [pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to find the appropriate PyTorch installation command for your system (CUDA version, CPU only, etc.).
    *   Run the PyTorch installation command *within the activated virtual environment*.
    *   Then re-run the `install.bat` script to install the remaining dependencies.

## Usage

1.  **Run the `launch.bat` script:**

    *   This script activates the virtual environment and starts the Gradio app.

2.  **Open the URL in your browser:**

    *   The app will typically be accessible at `http://127.0.0.1:7860` (or a similar address)

3.  **Enter your prompt, adjust settings, and click "Generate Image".**

## Configuration

*   **Model Folder:** The model is cached in the `models` folder.
*   **Output Folder:** Generated images are saved to the `output` folder.
*   You can change width and height but keep in mind the VRAM requirements. Smaller images will require less VRAM.

## Troubleshooting

*   **CUDA issues:** Ensure you have the correct CUDA drivers installed for your GPU. Check your PyTorch installation.
*   **Out of memory errors:** Reduce the image width and height, lower the inference steps, and ensure you're using CPU offloading. Close other applications that may be using GPU memory.
*   **Model loading errors:** Verify that the model name in `app.py` is correct and that you have a stable internet connection during the first run.

## License

[License]
