# Image Processing Application

A Python-based image processing application with a modern UI that supports various image processing algorithms including edge detection, filtering, and more.

## Prerequisites

- Python 3.11 or higher
- Windows OS (tested on Windows 10/11)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd improcess
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
.venv\Scripts\activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

### From Source
With the virtual environment activated:
```bash
python main.py
```

### Building Executable

1. Ensure you have all requirements installed and virtual environment activated
2. Run the build script:
```bash
build.bat
```
The executable will be created in `dist/improcess/` directory.

## Usage

1. Launch the application
2. Load an image by:
   - Dragging and dropping an image file
   - Using the "Open Image" button
3. Select an algorithm from the dropdown menu
4. Adjust parameters using the sliders
5. The processed image will update in real-time

## Dependencies

See [`requirements.txt`](requirements.txt) for a list of required packages.