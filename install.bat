@echo off
title KolorForge Kolors Web-UI app

echo Checking for existing virtual environment (venv)...
IF NOT EXIST venv (
    echo Creating Python virtual environment (venv)...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment. Please ensure Python is installed and in PATH.
        pause
        exit /b 1
    )
) ELSE (
    echo Virtual environment (venv) already exists. Skipping creation.
)

echo Activating virtual environment...
call venv\Scripts\activate
IF ERRORLEVEL 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch for NVIDIA GPU (CUDA 12.1)...
echo This is a common version for many modern NVIDIA GPUs like the RTX 3060.
echo If this fails, or if you have a different GPU/CUDA setup (e.g., older CUDA, AMD, or CPU only),
echo you may need to install PyTorch manually first by following instructions at:
echo https://pytorch.org/get-started/locally/
echo After manual PyTorch installation, you can re-run this script, and it
echo should skip this step if PyTorch is already detected with CUDA.
echo.

rem Check if torch with CUDA is already installed and a specific version (e.g. 2.3.1+cu121)
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__} with CUDA already installed.')" > nul 2>&1
IF ERRORLEVEL 0 (
    echo PyTorch with CUDA support already detected. Skipping PyTorch installation.
) ELSE (
    echo Attempting to install PyTorch with CUDA 12.1 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    IF ERRORLEVEL 1 (
        echo Failed to install PyTorch with CUDA 12.1.
        echo Please try installing PyTorch manually from https://pytorch.org/get-started/locally/
        echo then re-run this installer to get other dependencies.
        pause
        exit /b 1
    )
    echo PyTorch (CUDA 12.1) installation attempted.
)


echo.
echo Installing other dependencies from requirements.txt...
pip install --upgrade -r requirements.txt
IF ERRORLEVEL 1 (
    echo Failed to install other dependencies. Please check requirements.txt and your internet connection.
    pause
    exit /b 1
)

echo.
echo Installation complete.
echo.
echo --------------------------------------------------------------------
echo To run the app:
echo 1. (If not already) Open a new Command Prompt or PowerShell window.
echo 2. Navigate to this project directory.
echo 3. Activate the virtual environment: venv\Scripts\activate
echo 4. Then run the application: python app.py
echo (Or simply use launch.bat)
echo --------------------------------------------------------------------
echo.
pause
