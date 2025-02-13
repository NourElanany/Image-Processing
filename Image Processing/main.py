import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import importlib
import inspect
from typing import List, Callable, Dict, Final
from tkinterdnd2 import DND_FILES, TkinterDnD
import numpy as np
import threading

class ImageProcessorApp:
    def __init__(self):
        self.param_frame = None
        self.status_label = None
        self.progress_frame = None
        self.progress_bar = None
        self.sequence_label = None
        self.sequence_scroll = None
        self.sequence_frame = None
        self.clear_btn = None
        self.apply_btn = None
        self.middle_section = None
        self.open_btn = None
        self.save_btn = None
        self.right_buttons = None
        self.left_buttons = None
        self.top_control = None
        self.control_frame = None
        self.no_param_label = None
        self.param_scroll = None
        self.param_scroll = None
        self.tabview = None
        self.alg_container = None
        self.output_canvas = None
        self.right_frame = None
        self.input_canvas = None
        self.left_frame = None
        self.middle_frame = None
        self.image_loaded = False
        self.app = TkinterDnD.Tk()
        self.app.title("Image Processor")
        self.app.geometry("1400x900")

        # Theme settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.app.configure(bg='#2b2b2b')

        # Processing flag
        self.is_processing = False

        # Variables
        # Initialize image variables
        self.input_image = None  # Original loaded image
        self.current_image = None  # Current processed image
        self.output_image = None  # Final output image
        self.algorithm_sequence: List[Dict] = []
        self.algorithm_categories = {
            "Basic": ["Brightness", "Negative", "Rgb2Gray", "RGB2Binary", "Gray2Binary", "PointSharpening"],
            "Filters": ["MeanFilter", "MedianFilter", "MaxFilter", "MinFilter", "WeightFilter", "MidPointFilter"],
            "Edge Detection": ["SobelEdgeDetection", "RobertsEdgeDetection"],
            "Noise": [
                "GaussianNoise", "SaltAndPepperNoise", "UniformNoise",
                "RayleighNoise", "GammaNoise", "ExponentialNoise"
            ],
            "Frequency Domain": [
                "FourierTransform", "InverseFourierTransform",
                "IdealLowPassFilter", "IdealHighPassFilter",
                "ButterworthLowPassFilter", "ButterworthHighPassFilter",
                "GaussianLowPassFilter", "GaussianHighPassFilter"
            ],
            "Enhancement": ["HistogramEqualization", "ContrastStretching", "GammaCorrection", "Histogram"]
        }

        # Algorithm parameters dictionary
        self.algorithm_params = {
            "Brightness": {
                "factor": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.0, 3.0],
                    "step": 0.1
                }
            },
            "PointSharpening": {
                "factor": {
                    "type": "float",
                    "default": 1.5,
                    "range": [0.5, 3.0],
                    "step": 0.1
                }
            },
            "MidPointFilter": {
                "size": {
                    "type": "int",
                    "default": 3,
                    "range": (3, 9),
                    "step": 2
                }
            },
            "SobelEdgeDetection": {
                "threshold": {
                    "type": "int",
                    "default": 30,
                    "range": [0, 100],
                    "step": 1,
                },
            },
            "RobertsEdgeDetection": {
                "threshold": {
                    "type": "int",
                    "default": 30,
                    "range": [0, 100],
                    "step": 1,
                },
            },
            "GammaNoise": {
                "shape": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.1, 5.0],
                    "step": 0.1
                },
                "scale": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.1, 5.0],
                    "step": 0.1
                }
            },
            "RGB2Binary": {"threshold": {"type": "int", "default": 127, "range": (0, 255), "step": 1}},
            "Gray2Binary": {"threshold": {"type": "int", "default": 127, "range": (0, 255), "step": 1}},
            "GaussianNoise": {
                "mean": {
                    "type": "float",
                    "default": 0.0,
                    "range": [-0.2, 0.2],
                    "step": 0.01
                },
                "sigma": {
                    "type": "float",
                    "default": 0.1,
                    "range": [0.0, 0.5],
                    "step": 0.01
                }
            },
            "SaltAndPepperNoise": {
                "prob": {
                    "type": "float",
                    "default": 0.05,
                    "range": [0.0, 0.3],
                    "step": 0.01
                }
            },
            "UniformNoise": {
                "low": {
                    "type": "float",
                    "default": -0.2,
                    "range": [-0.5, 0.0],
                    "step": 0.01
                },
                "high": {
                    "type": "float",
                    "default": 0.2,
                    "range": [0.0, 0.5],
                    "step": 0.01
                }
            },
            "RayleighNoise": {
                "scale": {
                    "type": "float",
                    "default": 0.1,
                    "range": [0.0, 0.5],
                    "step": 0.01
                }
            },
            "MeanFilter": {"size": {"type": "int", "default": 3, "range": (3, 9), "step": 2}},
            "MedianFilter": {"size": {"type": "int", "default": 3, "range": (3, 9), "step": 2}},
            "MaxFilter": {"size": {"type": "int", "default": 3, "range": (3, 9), "step": 2}},
            "MinFilter": {"size": {"type": "int", "default": 3, "range": (3, 9), "step": 2}},
            "GaussianLowPassFilter": {"sigma": {"type": "float", "default": 1.0, "range": (0.1, 5.0), "step": 0.1}},
            "GaussianHighPassFilter": {"sigma": {"type": "float", "default": 1.0, "range": (0.1, 5.0), "step": 0.1}},
            "IdealLowPassFilter": {"cutoff": {"type": "int", "default": 30, "range": (1, 100), "step": 1}},
            "IdealHighPassFilter": {"cutoff": {"type": "int", "default": 30, "range": (1, 100), "step": 1}},
            "ButterworthLowPassFilter": {
                "cutoff": {"type": "int", "default": 30, "range": (1, 100), "step": 1},
                "order": {"type": "int", "default": 2, "range": (1, 5), "step": 1}
            },
            "ButterworthHighPassFilter": {
                "cutoff": {"type": "int", "default": 30, "range": (1, 100), "step": 1},
                "order": {"type": "int", "default": 2, "range": (1, 5), "step": 1}
            },
            "GammaCorrection": {"gamma": {"type": "float", "default": 1.0, "range": (0.1, 3.0), "step": 0.1}},
            "ContrastStretching": {
                "low_percentile": {
                    "type": "float",
                    "default": 2.0,
                    "range": [0.0, 20.0],
                    "step": 0.5
                },
                "high_percentile": {
                    "type": "float",
                    "default": 98.0,
                    "range": [80.0, 100.0],
                    "step": 0.5
                }
            }
        }

        # Current parameters for the selected algorithm
        self.current_params = {}

        # Message display variables
        self.message_label = None
        self.message_after_id = None

        self.setup_ui()
        self.load_algorithms()

    def setup_ui(self):
        # Main layout with three columns
        self.left_frame = ctk.CTkFrame(self.app, width=450, height=700)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.middle_frame = ctk.CTkFrame(self.app, width=450, height=700)
        self.middle_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.right_frame = ctk.CTkFrame(self.app, width=450, height=700)
        self.right_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        # Image display areas with labels
        self.input_canvas = None
        self.output_canvas = None
        self.setup_image_frame(self.left_frame, "Input Image", self.input_canvas)
        self.setup_image_frame(self.right_frame, "Output Image", self.output_canvas)

        # Algorithm selection in middle frame
        self.setup_algorithm_frame()

        # Control panel at bottom
        self.setup_control_panel()

        # Configure grid weights
        self.app.grid_rowconfigure(0, weight=1)
        self.app.grid_columnconfigure(0, weight=1)
        self.app.grid_columnconfigure(1, weight=1)
        self.app.grid_columnconfigure(2, weight=1)

    def setup_image_frame(self, parent, title, canvas):
        # Title
        title_label = ctk.CTkLabel(parent, text=title, font=("Arial", 16, "bold"))
        title_label.pack(pady=5)

        # Canvas
        canvas = tk.Canvas(parent, width=430, height=430, bg='#2b2b2b', highlightthickness=0)
        canvas.pack(pady=10)

        if title == "Input Image":
            self.input_canvas = canvas
            self.setup_drag_drop()
        else:
            self.output_canvas = canvas

    def setup_algorithm_frame(self):
        # Create a frame to hold both algorithm tabs and parameter section
        self.alg_container = ctk.CTkFrame(self.middle_frame)
        self.alg_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Title for algorithms
        title_label = ctk.CTkLabel(self.alg_container, text="Image Processing Algorithms",
                                   font=("Arial", 16, "bold"))
        title_label.pack(pady=5)

        # Create tabview for categories (reduced height to make room for parameters)
        self.tabview = ctk.CTkTabview(self.alg_container)
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs for each category
        for category in self.algorithm_categories:
            tab = self.tabview.add(category)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_columnconfigure(1, weight=1)

        # Parameters section
        self.setup_parameter_section()

    def setup_parameter_section(self):
        # Create frame for parameters
        self.param_frame = ctk.CTkFrame(self.middle_frame)
        self.param_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Title for parameters
        param_title = ctk.CTkLabel(self.param_frame, text="Algorithm Parameters",
                                   font=("Arial", 16, "bold"))
        param_title.pack(pady=5)

        # Scrollable frame for parameters
        self.param_scroll = ctk.CTkScrollableFrame(self.param_frame)
        self.param_scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Default message
        self.no_param_label = ctk.CTkLabel(self.param_scroll,
                                           text="Select algorithms to view parameters")
        self.no_param_label.pack(pady=10)

    def update_parameter_section(self):
        # Clear previous parameters
        for widget in self.param_scroll.winfo_children():
            widget.destroy()

        if not self.algorithm_sequence:
            self.no_param_label = ctk.CTkLabel(self.param_scroll,
                                               text="Select algorithms to view parameters")
            self.no_param_label.pack(pady=10)
            return

        # Create sections for each algorithm in sequence
        for alg in self.algorithm_sequence:
            algorithm_name = alg["name"]
            params = self.algorithm_params.get(algorithm_name)

            if not params:
                continue

            # Create a frame for this algorithm's parameters
            alg_frame = ctk.CTkFrame(self.param_scroll)
            alg_frame.pack(fill=tk.X, padx=5, pady=5)

            # Algorithm name label
            alg_label = ctk.CTkLabel(alg_frame, text=algorithm_name,
                                     font=("Arial", 12, "bold"))
            alg_label.pack(pady=5)

            # Create widgets for each parameter
            for param_name, param_info in params.items():
                # Parameter frame
                param_frame = ctk.CTkFrame(alg_frame)
                param_frame.pack(fill=tk.X, padx=5, pady=2)

                # Label
                label = ctk.CTkLabel(param_frame, text=f"{param_name}:")
                label.pack(side=tk.LEFT, padx=5)

                # Create appropriate widget based on parameter type
                if param_info["type"] in ["float", "int"]:
                    value = ctk.CTkSlider(
                        param_frame,
                        from_=int(param_info["range"][0]),
                        to=int(param_info["range"][1]),
                        number_of_steps=int((param_info["range"][1] - param_info["range"][0]) / param_info["step"])
                    )
                    # Set value from current parameters or default
                    current_value = alg["params"].get(param_name, param_info["default"])
                    value.set(current_value)
                    value.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

                    # Value label
                    value_label = ctk.CTkLabel(param_frame, text=str(current_value))
                    value_label.pack(side=tk.LEFT, padx=5)

                    # Update function
                    def update_value(val, label=value_label, param=param_name,
                                     alg_item=alg, param_type=param_info["type"]):
                        if param_type == "int":
                            val = int(val)
                        label.configure(text=f"{val:.2f}" if param_type == "float" else str(val))
                        
                        # Update the parameters in both places
                        alg_item["params"][param] = val
                        self.current_params[alg_item["name"]][param] = val

                    value.configure(command=update_value)

            # Add separator
            separator = ctk.CTkFrame(self.param_scroll, height=1, fg_color="gray30")
            separator.pack(fill=tk.X, padx=15, pady=5)

    def setup_control_panel(self):
        # Control panel with two rows
        self.control_frame = ctk.CTkFrame(self.app)
        self.control_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Message label for notifications
        self.message_label = ctk.CTkLabel(self.control_frame, text="", text_color="orange")
        self.message_label.pack(pady=(0, 5))

        # Top row: Buttons and sequence display
        self.top_control = ctk.CTkFrame(self.control_frame)
        self.top_control.pack(fill=tk.X, padx=5, pady=5)

        # Left section for file buttons
        self.left_buttons = ctk.CTkFrame(self.top_control)
        self.left_buttons.pack(side=tk.LEFT, padx=5)

        self.open_btn = ctk.CTkButton(self.left_buttons, text="Open Image", command=self.open_image)
        self.open_btn.pack(side=tk.LEFT, padx=2)

        self.save_btn = ctk.CTkButton(
            self.left_buttons, text="Save Image", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=2)

        # Add Medical Enhancement button (disabled by default)
        self.medical_btn = ctk.CTkButton(
            self.left_buttons, 
            text="Medical Enhancement",
            command=self.show_medical_enhancement,
            state="disabled"  # Initially disabled
        )
        self.medical_btn.pack(side=tk.LEFT, padx=2)

        # Right section for control buttons
        self.right_buttons = ctk.CTkFrame(self.top_control)
        self.right_buttons.pack(side=tk.RIGHT, padx=5)

        self.clear_btn = ctk.CTkButton(self.right_buttons, text="Clear Sequence",
                                       command=self.clear_sequence)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        self.apply_btn = ctk.CTkButton(self.right_buttons, text="Apply",
                                       command=self.start_processing, state="disabled")
        self.apply_btn.pack(side=tk.LEFT, padx=2)

        # Middle section with fixed proportions
        self.middle_section = ctk.CTkFrame(self.top_control)
        self.middle_section.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Configure grid columns with weights
        self.middle_section.grid_columnconfigure(0, weight=2)  # Sequence gets more space
        self.middle_section.grid_columnconfigure(1, weight=1)  # Progress gets less space

        # Sequence display frame with fixed height
        self.sequence_frame = ctk.CTkFrame(self.middle_section, fg_color='#1f1f1f')
        self.sequence_frame.grid(row=0, column=0, sticky="ew", padx=2)
        self.sequence_frame.configure(height=60)  # Fixed height
        self.sequence_frame.grid_propagate(False)  # Prevent frame from resizing

        # Scrollable frame for sequence
        self.sequence_scroll = ctk.CTkScrollableFrame(
            self.sequence_frame,
            fg_color='#1f1f1f',
            height=50,  # Slightly less than parent to ensure scrollbar shows
            orientation="horizontal"  # Horizontal scrolling
        )
        self.sequence_scroll.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.sequence_label = ctk.CTkLabel(
            self.sequence_scroll,
            text="No algorithms selected",
            fg_color='#1f1f1f'
        )
        self.sequence_label.pack(side=tk.LEFT, padx=5, fill=tk.Y, expand=True)

        # Progress frame
        self.progress_frame = ctk.CTkFrame(self.middle_section)
        self.progress_frame.grid(row=0, column=1, sticky="ew", padx=2)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(self.progress_frame, text="")

    def update_sequence_display(self):
        if not self.algorithm_sequence:
            self.sequence_label.configure(text="No algorithms selected")
        else:
            sequence_text = " → ".join([alg["name"] for alg in self.algorithm_sequence])
            # Update label with arrow separator
            self.sequence_label.configure(text=sequence_text)
            # Force update of scrollable frame
            self.sequence_scroll.update()

    def show_progress(self):
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)
        self.status_label.pack(pady=2)

    def hide_progress(self):
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()

    def start_processing(self):
        if self.is_processing:
            return

        if not self.input_image or not self.algorithm_sequence:
            return

        # Disable buttons during processing
        self.apply_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")

        # Show progress bar
        self.progress_bar.set(0)
        self.show_progress()
        self.status_label.configure(text="Processing...")
        self.is_processing = True

        # Start processing thread
        thread = threading.Thread(target=self.process_image_thread)
        thread.daemon = True
        thread.start()

        # Start checking progress
        self.app.after(100, self.check_processing)

    def process_image_thread(self):
        try:
            processed_image = self.input_image.copy()
            total_steps = len(self.algorithm_sequence)

            # Track if we're in frequency domain
            in_frequency_domain = False

            for i, alg in enumerate(self.algorithm_sequence):
                # Update progress
                progress = (i / total_steps)
                self.app.after(0, self.update_progress, progress, f"Applying {alg['name']}...")

                try:
                    # Convert to numpy array if it's a PIL Image
                    if isinstance(processed_image, Image.Image):
                        if processed_image.mode != 'RGB':
                            processed_image = processed_image.convert('RGB')
                        processed_image = np.array(processed_image)

                    # Apply the algorithm with parameters
                    if alg["name"] == "FourierTransform":
                        # Convert to grayscale if needed
                        if len(processed_image.shape) == 3:
                            processed_image = np.dot(processed_image[..., :3], [0.2989, 0.5870, 0.1140])

                        display_spectrum, freq_data = alg["function"](processed_image, **alg.get("params", {}))
                        processed_image = freq_data  # Store frequency domain data
                        in_frequency_domain = True

                        # Update display with magnitude spectrum
                        self.app.after(0, self.display_image, Image.fromarray(display_spectrum), self.output_canvas)

                    elif alg["name"] == "InverseFourierTransform":
                        if not in_frequency_domain:
                            self.app.after(0, self.show_error,
                                         f"Error: Must apply Fourier Transform before {alg['name']}")
                            return
                        processed_image = alg["function"](processed_image)  # Get spatial domain image
                        in_frequency_domain = False

                    elif alg["name"] in ["IdealLowPassFilter", "IdealHighPassFilter",
                                       "ButterworthLowPassFilter", "ButterworthHighPassFilter",
                                       "GaussianLowPassFilter", "GaussianHighPassFilter"]:
                        if not in_frequency_domain:
                            self.app.after(0, self.show_error,
                                         f"Error: Must apply Fourier Transform before {alg['name']}")
                            return
                        
                        # Apply filter to frequency domain data
                        filtered_freq = alg["function"](processed_image, **alg.get("params", {}))
                        processed_image = filtered_freq  # Keep in frequency domain
                        
                        # Update display with new magnitude spectrum
                        magnitude_spectrum = np.log(np.abs(filtered_freq) + 1)
                        display_spectrum = ((magnitude_spectrum - magnitude_spectrum.min()) * 255
                                          / (magnitude_spectrum.max() - magnitude_spectrum.min()))
                        self.app.after(0, self.display_image, Image.fromarray(display_spectrum.astype(np.uint8)), self.output_canvas)

                    else:
                        if in_frequency_domain:
                            self.app.after(0, self.show_error,
                                         f"Error: Must apply Inverse Fourier Transform before {alg['name']}")
                            return
                        processed_image = alg["function"](processed_image, **alg.get("params", {}))

                    # Convert back to PIL Image if not in frequency domain
                    if not in_frequency_domain and isinstance(processed_image, np.ndarray):
                        # Ensure the array is uint8
                        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
                        processed_image = Image.fromarray(processed_image)

                except Exception as e:
                    self.app.after(0, self.show_error, f"Error applying {alg['name']}: {str(e)}")
                    return

            # Update progress to 100%
            self.app.after(0, self.update_progress, 1.0, "Processing complete!")

            # Update output image
            self.output_image = processed_image
            self.app.after(0, self.display_image, processed_image, self.output_canvas)

        finally:
            # Reset processing state
            self.is_processing = False

    def update_progress(self, progress, status):
        self.progress_bar.set(progress)
        self.status_label.configure(text=status)

    def show_error(self, error_message):
        self.status_label.configure(text=error_message)
        self.hide_progress()
        self.enable_buttons()

    def check_processing(self):
        if self.is_processing:
            self.app.after(100, self.check_processing)
        else:
            self.enable_buttons()
            self.app.after(2000, self.hide_progress)

    def enable_buttons(self):
        self.apply_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        self.open_btn.configure(state="normal")
        self.save_btn.configure(state="normal")

    def setup_drag_drop(self):
        self.input_canvas.drop_target_register(DND_FILES)
        self.input_canvas.dnd_bind('<<Drop>>', self.handle_drop)

        # Add drop zone indication
        self.input_canvas.create_text(
            215, 215,
            text="Drag and Drop Image Here\nor\nClick 'Open Image'",
            fill='#666666',
            justify=tk.CENTER,
            font=('Arial', 14)
        )

    def handle_drop(self, event):
        file_path = event.data
        if file_path.startswith('{'):
            file_path = file_path[1:]
        if file_path.endswith('}'):
            file_path = file_path[:-1]
        self.load_image(file_path)
        self.medical_btn.configure(state="normal")  # Enable medical button

    def load_algorithms(self):
        alg_path = os.path.join(os.path.dirname(__file__), 'alg')
        loaded_algorithms = {}

        # Load all algorithms first
        for file in os.listdir(alg_path):
            if file.endswith('.py') and not file.startswith('__'):
                module_name = file[:-3]
                try:
                    module = importlib.import_module(f'alg.{module_name}')
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and name == module_name:
                            loaded_algorithms[module_name] = obj
                except Exception as e:
                    print(f"Error loading {module_name}: {e}")

        # Add algorithms to their respective category tabs
        for category, algs in self.algorithm_categories.items():
            tab = self.tabview.tab(category)
            for i, alg_name in enumerate(algs):
                if alg_name in loaded_algorithms:
                    row = i // 2
                    col = i % 2
                    btn = ctk.CTkButton(tab, text=alg_name,
                                        command=lambda n=alg_name, f=loaded_algorithms[alg_name]:
                                        self.add_to_sequence(n, f))
                    btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

    def clear_sequence(self):
        self.algorithm_sequence.clear()
        self.update_sequence_display()
        self.update_parameter_section()

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        if file_path:
            self.load_image(file_path)
            self.medical_btn.configure(state="normal")  # Enable medical button

    def save_image(self):
        if not self.output_image:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"),
                       ("JPEG files", "*.jpg"),
                       ("All files", "*.*")]
        )
        if file_path:
            self.output_image.save(file_path)

    def load_image(self, file_path):
        try:
            image = Image.open(file_path)
            # Resize image to fit canvas while maintaining aspect ratio
            display_size = (430, 430)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)

            self.input_image = image
            self.display_image(image, self.input_canvas)
            self.output_image = None
            self.output_canvas.delete("all")

            # Add placeholder text to output canvas
            self.output_canvas.create_text(
                215, 215,
                text="Processed image will appear here",
                fill='#666666',
                font=('Arial', 14)
            )
            self.image_loaded = True
            self.apply_btn.configure(state="normal")
            print("Image Loaded")
        except Exception as e:
            self.image_loaded = False
            self.apply_btn.configure(state="disabled")
            print(f"Error loading image: {e}")
        else:
            print("Image Loaded")
            self.image_loaded = True

    @staticmethod
    def display_image(image: Image.Image, canvas: tk.Canvas):
        if image is None:
            return
            
        # Convert NumPy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
            
        # Create PhotoImage
        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.image = photo  # Keep a reference
        canvas.create_image(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            image=photo,
            anchor="center"
        )

    def add_to_sequence(self, name: str, func: Callable):
        # Check if this would create a duplicate in sequence
        if self.algorithm_sequence and self.algorithm_sequence[-1]["name"] == name:
            self.show_message(f"Cannot add duplicate {name} in sequence")
            return

        # Initialize parameters if not exists
        if name not in self.current_params:
            self.current_params[name] = {}
            # Set default values from algorithm_params
            if name in self.algorithm_params:
                for param_name, param_info in self.algorithm_params[name].items():
                    self.current_params[name][param_name] = param_info["default"]

        # Create a copy of current parameters for this algorithm
        params = self.current_params[name].copy()

        # Add to sequence
        self.algorithm_sequence.append({
            "name": name,
            "function": func,
            "params": params
        })
        self.update_sequence_display()

        # Update parameter section to show all algorithms
        self.update_parameter_section()

    def show_message(self, message: str, duration: int = 3000):
        """Show a temporary message to the user."""
        # Cancel any existing message timer
        if self.message_after_id:
            self.app.after_cancel(self.message_after_id)

        # Show new message
        self.message_label.configure(text=message)

        # Schedule message removal
        self.message_after_id = self.app.after(duration, lambda: self.message_label.configure(text=""))

    def show_medical_enhancement(self):
        """Show the medical enhancement window"""
        if not hasattr(self, 'medical_gui'):
            from medical_gui import MedicalEnhancementGUI
            self.medical_gui = MedicalEnhancementGUI(self)
        self.medical_gui.show()

    def get_algorithm_module(self, name: str):
        """Get algorithm module by name - used by medical enhancement GUI"""
        try:
            module = importlib.import_module(f"alg.{name}")
            return getattr(module, name)
        except (ImportError, AttributeError):
            return None

    def run(self):
        self.app.mainloop()


if __name__ == '__main__':
    app = ImageProcessorApp()
    app.run()
