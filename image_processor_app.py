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
        # Initialize all UI components to None first
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
        self.tabview = None
        self.alg_container = None
        self.output_canvas = None
        self.right_frame = None
        self.input_canvas = None
        self.left_frame = None
        self.middle_frame = None
        self.dashboard_frame = None
        self.traditional_btn = None
        self.neural_btn = None
        
        # Initialize application state
        self.image_loaded = False
        self.app = TkinterDnD.Tk()
        self.app.title("Image Processor - Enhanced Dashboard")
        self.app.geometry("1500x1000")  # Increased size for better layout
        self.app.minsize(1200, 800)  # Set minimum size

        # Theme settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.app.configure(bg='#2b2b2b')

        # Processing flag
        self.is_processing = False

        # Current mode: 'traditional' or 'neural'
        self.current_mode = 'traditional'

        # Variables
        # Initialize image variables
        self.input_image = None  # Original loaded image
        self.current_image = None  # Current processed image
        self.output_image = None  # Final output image
        self.algorithm_sequence: List[Dict] = []
        
        # Traditional image processing algorithms
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
        
        # Neural Network algorithms
        self.neural_categories = {
            "Feature Detection": ["SIFTFeatureDetection", "ORBFeatureDetection", "SURFFeatureDetection"],
            "Corner Detection": ["HarrisCornerDetection", "FASTCornerDetection", "GoodFeaturesToTrack"],
            "Feature Matching": ["FeatureMatching", "FeatureMatchingHomography"],
            "Advanced Features": ["BRIEFFeatureDetection"]
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
            },
            # Neural Network algorithm parameters
            "SIFTFeatureDetection": {
                "nfeatures": {"type": "int", "default": 0, "range": (0, 1000), "step": 50},
                "contrastThreshold": {"type": "float", "default": 0.04, "range": (0.01, 0.1), "step": 0.01},
                "edgeThreshold": {"type": "int", "default": 10, "range": (5, 20), "step": 1},
                "sigma": {"type": "float", "default": 1.6, "range": (1.0, 3.0), "step": 0.1}
            },
            "ORBFeatureDetection": {
                "nfeatures": {"type": "int", "default": 500, "range": (100, 2000), "step": 100},
                "scaleFactor": {"type": "float", "default": 1.2, "range": (1.1, 2.0), "step": 0.1},
                "nlevels": {"type": "int", "default": 8, "range": (3, 15), "step": 1},
                "edgeThreshold": {"type": "int", "default": 31, "range": (10, 50), "step": 1},
                "fastThreshold": {"type": "int", "default": 20, "range": (5, 50), "step": 1}
            },
            "HarrisCornerDetection": {
                "blockSize": {"type": "int", "default": 2, "range": (2, 10), "step": 1},
                "ksize": {"type": "int", "default": 3, "range": (3, 7), "step": 2},
                "k": {"type": "float", "default": 0.04, "range": (0.01, 0.1), "step": 0.01},
                "threshold": {"type": "float", "default": 0.01, "range": (0.001, 0.1), "step": 0.001}
            },
            "FASTCornerDetection": {
                "threshold": {"type": "int", "default": 10, "range": (5, 50), "step": 1}
            },
            "GoodFeaturesToTrack": {
                "maxCorners": {"type": "int", "default": 100, "range": (10, 500), "step": 10},
                "qualityLevel": {"type": "float", "default": 0.01, "range": (0.001, 0.1), "step": 0.001},
                "minDistance": {"type": "int", "default": 10, "range": (1, 50), "step": 1},
                "blockSize": {"type": "int", "default": 3, "range": (3, 9), "step": 2}
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
        """Setup the main UI with proper layout organization"""
        # Configure main window grid - 4 rows for better organization
        self.app.grid_rowconfigure(0, weight=0)  # Dashboard row (fixed height)
        self.app.grid_rowconfigure(1, weight=1)  # Main content row (expandable)
        self.app.grid_rowconfigure(2, weight=0)  # Control panel row (fixed height)
        self.app.grid_rowconfigure(3, weight=0)  # Status row (fixed height)
        self.app.grid_columnconfigure(0, weight=1)
        self.app.grid_columnconfigure(1, weight=1)
        self.app.grid_columnconfigure(2, weight=1)

        # Setup dashboard at the very top
        self.setup_dashboard()
        
        # Setup main content area (images and algorithms)
        self.setup_main_content()
        
        # Setup control panel
        self.setup_control_panel()
        
        # Setup status bar
        self.setup_status_bar()

    def setup_dashboard(self):
        """Setup the main dashboard with mode selection buttons at the top"""
        self.dashboard_frame = ctk.CTkFrame(self.app, height=100)
        self.dashboard_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="ew")
        self.dashboard_frame.grid_propagate(False)
        
        # Title
        title_label = ctk.CTkLabel(
            self.dashboard_frame, 
            text="Image Processing Dashboard", 
            font=ctk.CTkFont(family="Arial", size=24, weight="bold")
        )
        title_label.pack(pady=(10, 5))
        
        # Mode selection buttons frame
        button_frame = ctk.CTkFrame(self.dashboard_frame)
        button_frame.pack(pady=(5, 10))
        
        self.traditional_btn = ctk.CTkButton(
            button_frame, 
            text="Traditional Image Processing",
            command=self.switch_to_traditional,
            width=280,
            height=40,
            font=ctk.CTkFont(family="Arial", size=14, weight="bold")
        )
        self.traditional_btn.pack(side=tk.LEFT, padx=10)
        
        self.neural_btn = ctk.CTkButton(
            button_frame, 
            text="Neural Network Algorithms",
            command=self.switch_to_neural,
            width=280,
            height=40,
            font=ctk.CTkFont(family="Arial", size=14, weight="bold")
        )
        self.neural_btn.pack(side=tk.LEFT, padx=10)
        
        # Set initial button states
        self.update_dashboard_buttons()

    def setup_main_content(self):
        """Setup the main content area with three columns for images and algorithms"""
        # Left frame - Input Image
        self.left_frame = ctk.CTkFrame(self.app)
        self.left_frame.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="nsew")
        self.left_frame.grid_propagate(False)

        # Middle frame - Algorithm Selection and Parameters
        self.middle_frame = ctk.CTkFrame(self.app)
        self.middle_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.middle_frame.grid_propagate(False)

        # Right frame - Output Image
        self.right_frame = ctk.CTkFrame(self.app)
        self.right_frame.grid(row=1, column=2, padx=(5, 10), pady=5, sticky="nsew")
        self.right_frame.grid_propagate(False)

        # Setup image display areas
        self.setup_image_displays()
        
        # Setup algorithm selection area
        self.setup_algorithm_frame()

    def setup_image_displays(self):
        """Setup input and output image display areas"""
        # Input image setup
        input_title = ctk.CTkLabel(
            self.left_frame, 
            text="Input Image", 
            font=ctk.CTkFont(family="Arial", size=18, weight="bold")
        )
        input_title.pack(pady=(10, 5))

        self.input_canvas = tk.Canvas(
            self.left_frame, 
            width=450, 
            height=450, 
            bg='#2b2b2b', 
            highlightthickness=2,
            highlightcolor="#1f538d",
            highlightbackground="#404040"
        )
        self.input_canvas.pack(pady=10, padx=10)
        self.setup_drag_drop()

        # Output image setup
        output_title = ctk.CTkLabel(
            self.right_frame, 
            text="Output Image", 
            font=ctk.CTkFont(family="Arial", size=18, weight="bold")
        )
        output_title.pack(pady=(10, 5))

        self.output_canvas = tk.Canvas(
            self.right_frame, 
            width=450, 
            height=450, 
            bg='#2b2b2b', 
            highlightthickness=2,
            highlightcolor="#1f538d",
            highlightbackground="#404040"
        )
        self.output_canvas.pack(pady=10, padx=10)

        # Add placeholder text to output canvas
        self.output_canvas.create_text(
            225, 225,
            text="Processed image will appear here",
            fill='#666666',
            font=('Arial', 14),
            justify=tk.CENTER
        )

    def setup_control_panel(self):
        """Setup control panel with file operations and processing controls"""
        self.control_frame = ctk.CTkFrame(self.app, height=120)
        self.control_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        self.control_frame.grid_propagate(False)
        
        # File operations frame
        file_frame = ctk.CTkFrame(self.control_frame)
        file_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        
        file_label = ctk.CTkLabel(file_frame, text="File Operations", font=ctk.CTkFont(size=12, weight="bold"))
        file_label.pack(pady=(5, 10))
        
        self.open_btn = ctk.CTkButton(
            file_frame, 
            text="Open Image", 
            command=self.open_image,
            width=120,
            height=35
        )
        self.open_btn.pack(pady=2)

        self.save_btn = ctk.CTkButton(
            file_frame, 
            text="Save Image", 
            command=self.save_image,
            width=120,
            height=35
        )
        self.save_btn.pack(pady=2)

        # Algorithm sequence display (middle section)
        sequence_frame = ctk.CTkFrame(self.control_frame)
        sequence_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        sequence_label = ctk.CTkLabel(sequence_frame, text="Algorithm Sequence", font=ctk.CTkFont(size=12, weight="bold"))
        sequence_label.pack(pady=(5, 10))
        
        # Sequence display with horizontal scroll
        self.sequence_scroll = ctk.CTkScrollableFrame(
            sequence_frame,
            orientation="horizontal",
            height=60,
            fg_color='#1f1f1f'
        )
        self.sequence_scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.sequence_label = ctk.CTkLabel(
            self.sequence_scroll,
            text="No algorithms selected",
            font=ctk.CTkFont(size=11)
        )
        self.sequence_label.pack(side=tk.LEFT, padx=5)

        # Processing controls frame
        process_frame = ctk.CTkFrame(self.control_frame)
        process_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        process_label = ctk.CTkLabel(process_frame, text="Processing", font=ctk.CTkFont(size=12, weight="bold"))
        process_label.pack(pady=(5, 10))

        self.clear_btn = ctk.CTkButton(
            process_frame, 
            text="Clear Sequence",
            command=self.clear_sequence,
            width=120,
            height=35
        )
        self.clear_btn.pack(pady=2)

        self.apply_btn = ctk.CTkButton(
            process_frame, 
            text="Apply Algorithms",
            command=self.start_processing, 
            state="disabled",
            width=120,
            height=35
        )
        self.apply_btn.pack(pady=2)

    def setup_status_bar(self):
        """Setup status bar with progress indicator and messages"""
        status_frame = ctk.CTkFrame(self.app, height=60)
        status_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=(5, 10), sticky="ew")
        status_frame.grid_propagate(False)
        
        # Message label for notifications
        self.message_label = ctk.CTkLabel(
            status_frame, 
            text="Ready", 
            text_color="#4CAF50",
            font=ctk.CTkFont(size=12)
        )
        self.message_label.pack(side=tk.LEFT, padx=20, pady=15)

        # Progress section
        progress_container = ctk.CTkFrame(status_frame)
        progress_container.pack(side=tk.RIGHT, padx=20, pady=10, fill=tk.Y)
        
        self.status_label = ctk.CTkLabel(
            progress_container, 
            text="",
            font=ctk.CTkFont(size=10)
        )
        
        self.progress_bar = ctk.CTkProgressBar(progress_container, width=200)
        self.progress_bar.set(0)

    def update_dashboard_buttons(self):
        """Update dashboard button appearance based on current mode"""
        if self.current_mode == 'traditional':
            self.traditional_btn.configure(fg_color=("#1f538d", "#14375e"))  # Active color
            self.neural_btn.configure(fg_color=("#565b5e", "#343638"))  # Inactive color
        else:
            self.traditional_btn.configure(fg_color=("#565b5e", "#343638"))  # Inactive color
            self.neural_btn.configure(fg_color=("#1f538d", "#14375e"))  # Active color

    def switch_to_traditional(self):
        """Switch to traditional image processing mode"""
        self.current_mode = 'traditional'
        self.clear_sequence()
        self.setup_algorithm_frame()
        self.load_algorithms()
        self.update_dashboard_buttons()
        self.show_message("Switched to Traditional Image Processing Mode", "#4CAF50")

    def switch_to_neural(self):
        """Switch to neural network algorithms mode"""
        self.current_mode = 'neural'
        self.clear_sequence()
        self.setup_algorithm_frame()
        self.load_algorithms()
        self.update_dashboard_buttons()
        self.show_message("Switched to Neural Network Algorithms Mode", "#4CAF50")

    def setup_algorithm_frame(self):
        """Setup algorithm selection and parameter frames in the middle column"""
        # Clear existing algorithm container
        if hasattr(self, 'alg_container') and self.alg_container:
            self.alg_container.destroy()
            
        # Create main container for algorithms and parameters
        self.alg_container = ctk.CTkFrame(self.middle_frame)
        self.alg_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title for current mode
        mode_title = "Traditional Image Processing" if self.current_mode == 'traditional' else "Neural Network Algorithms"
        title_label = ctk.CTkLabel(
            self.alg_container, 
            text=mode_title,
            font=ctk.CTkFont(family="Arial", size=16, weight="bold")
        )
        title_label.pack(pady=(10, 5))

        # Create tabview for algorithm categories (50% of space)
        self.tabview = ctk.CTkTabview(self.alg_container, height=300)
        self.tabview.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Create tabs for each category based on current mode
        categories = self.algorithm_categories if self.current_mode == 'traditional' else self.neural_categories
        self.current_categories = categories
        for category in categories:
            tab = self.tabview.add(category)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_columnconfigure(1, weight=1)

        # Parameters section (50% of space)
        self.setup_parameter_section()

    def setup_parameter_section(self):
        """Setup parameter section below the algorithm tabs"""
        # Create frame for parameters
        param_container = ctk.CTkFrame(self.alg_container)
        param_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Title for parameters
        param_title = ctk.CTkLabel(
            param_container, 
            text="Algorithm Parameters",
            font=ctk.CTkFont(family="Arial", size=14, weight="bold")
        )
        param_title.pack(pady=(10, 5))

        # Scrollable frame for parameters
        self.param_scroll = ctk.CTkScrollableFrame(param_container, height=250)
        self.param_scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Default message
        self.no_param_label = ctk.CTkLabel(
            self.param_scroll,
            text="Select algorithms to view and adjust parameters",
            font=ctk.CTkFont(size=12)
        )
        self.no_param_label.pack(pady=20)

    def update_parameter_section(self):
        """Update parameter section with controls for selected algorithms"""
        # Clear previous parameters
        for widget in self.param_scroll.winfo_children():
            widget.destroy()

        if not self.algorithm_sequence:
            self.no_param_label = ctk.CTkLabel(
                self.param_scroll,
                text="Select algorithms to view and adjust parameters",
                font=ctk.CTkFont(size=12)
            )
            self.no_param_label.pack(pady=20)
            return

        # Create sections for each algorithm in sequence
        for i, alg in enumerate(self.algorithm_sequence):
            algorithm_name = alg["name"]
            params = self.algorithm_params.get(algorithm_name)

            if not params:
                continue

            # Create a frame for this algorithm's parameters
            alg_frame = ctk.CTkFrame(self.param_scroll)
            alg_frame.pack(fill=tk.X, padx=5, pady=5)

            # Algorithm name label with index
            alg_label = ctk.CTkLabel(
                alg_frame, 
                text=f"{i+1}. {algorithm_name}",
                font=ctk.CTkFont(family="Arial", size=12, weight="bold")
            )
            alg_label.pack(pady=(10, 5))

            # Create widgets for each parameter
            for param_name, param_info in params.items():
                # Parameter frame
                param_frame = ctk.CTkFrame(alg_frame)
                param_frame.pack(fill=tk.X, padx=10, pady=3)

                # Label
                label = ctk.CTkLabel(
                    param_frame, 
                    text=f"{param_name}:",
                    font=ctk.CTkFont(size=11)
                )
                label.pack(side=tk.LEFT, padx=(10, 5), pady=5)

                # Create appropriate widget based on parameter type
                if param_info["type"] in ["float", "int"]:
                    # Current value from parameters or default
                    current_value = alg["params"].get(param_name, param_info["default"])
                    
                    # Value display label
                    value_label = ctk.CTkLabel(
                        param_frame, 
                        text=f"{current_value:.2f}" if param_info["type"] == "float" else str(current_value),
                        font=ctk.CTkFont(size=11, weight="bold"),
                        width=60
                    )
                    value_label.pack(side=tk.RIGHT, padx=(5, 10), pady=5)

                    # Slider
                    slider = ctk.CTkSlider(
                        param_frame,
                        from_=param_info["range"][0],
                        to=param_info["range"][1],
                        number_of_steps=int((param_info["range"][1] - param_info["range"][0]) / param_info["step"])
                    )
                    slider.set(current_value)
                    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

                    # Update function
                    def update_value(val, label=value_label, param=param_name,
                                     alg_item=alg, param_type=param_info["type"]):
                        if param_type == "int":
                            val = int(val)
                        label.configure(text=f"{val:.2f}" if param_type == "float" else str(val))
                        
                        # Update the parameters in both places
                        alg_item["params"][param] = val
                        if alg_item["name"] not in self.current_params:
                            self.current_params[alg_item["name"]] = {}
                        self.current_params[alg_item["name"]][param] = val

                    slider.configure(command=update_value)

    def setup_drag_drop(self):
        """Setup drag and drop functionality for input canvas"""
        self.input_canvas.drop_target_register(DND_FILES)
        self.input_canvas.dnd_bind('<<Drop>>', self.handle_drop)

        # Add drop zone indication
        self.input_canvas.create_text(
            225, 225,
            text="Drag and Drop Image Here\nor\nClick 'Open Image'",
            fill='#666666',
            justify=tk.CENTER,
            font=('Arial', 14)
        )

    def handle_drop(self, event):
        """Handle dropped files"""
        file_path = event.data
        if file_path.startswith('{'):
            file_path = file_path[1:]
        if file_path.endswith('}'):
            file_path = file_path[:-1]
        self.load_image(file_path)

    def load_algorithms(self):
        """Load algorithm functions from files and add to UI"""
        alg_path = os.path.join(os.path.dirname(__file__), 'algorithm_image_processing')
        loaded_algorithms = {}

        # Load all algorithms first
        try:
            for file in os.listdir(alg_path):
                if file.endswith('.py') and not file.startswith('__'):
                    module_name = file[:-3]
                    try:
                        module = importlib.import_module(f'algorithm_image_processing.{module_name}')
                        for name, obj in inspect.getmembers(module):
                            if inspect.isfunction(obj) and name == module_name:
                                loaded_algorithms[module_name] = obj
                    except Exception as e:
                        print(f"Error loading {module_name}: {e}")
        except FileNotFoundError:
            print("Algorithm directory not found. Please ensure the 'algorithm_image_processing' directory exists.")
            return

        # Add algorithms to their respective category tabs
        categories = self.algorithm_categories if self.current_mode == 'traditional' else self.neural_categories
        for category, algs in categories.items():
            tab = self.tabview.tab(category)
            for i, alg_name in enumerate(algs):
                if alg_name in loaded_algorithms:
                    row = i // 2
                    col = i % 2
                    btn = ctk.CTkButton(
                        tab, 
                        text=alg_name,
                        command=lambda n=alg_name, f=loaded_algorithms[alg_name]: self.add_to_sequence(n, f),
                        height=35,
                        font=ctk.CTkFont(size=10)
                    )
                    btn.grid(row=row, column=col, padx=5, pady=3, sticky="ew")

    def add_to_sequence(self, name: str, func: Callable):
        """Add algorithm to processing sequence"""
        # Check if this would create a duplicate in sequence
        if self.algorithm_sequence and self.algorithm_sequence[-1]["name"] == name:
            self.show_message(f"Cannot add duplicate {name} in sequence", "#FF9800")
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
        self.update_parameter_section()
        
        # Enable apply button if image is loaded
        if self.input_image:
            self.apply_btn.configure(state="normal")
            
        self.show_message(f"Added {name} to sequence", "#4CAF50")

    def clear_sequence(self):
        """Clear the algorithm sequence"""
        self.algorithm_sequence.clear()
        self.update_sequence_display()
        self.update_parameter_section()
        self.apply_btn.configure(state="disabled")
        self.show_message("Sequence cleared", "#2196F3")

    def update_sequence_display(self):
        """Update the sequence display in control panel"""
        if not self.algorithm_sequence:
            self.sequence_label.configure(text="No algorithms selected")
        else:
            sequence_text = " → ".join([f"{i+1}. {alg['name']}" for i, alg in enumerate(self.algorithm_sequence)])
            self.sequence_label.configure(text=sequence_text)
            # Force update of scrollable frame
            self.sequence_scroll.update()

    def show_progress(self):
        """Show progress bar and status"""
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)
        self.status_label.pack(pady=2)

    def hide_progress(self):
        """Hide progress bar and status"""
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()

    def open_image(self):
        """Open image file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_image(file_path)

    def save_image(self):
        """Save output image"""
        if not self.output_image:
            self.show_message("No processed image to save", "#FF5722")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                self.output_image.save(file_path)
                self.show_message(f"Image saved successfully: {os.path.basename(file_path)}", "#4CAF50")
            except Exception as e:
                self.show_message(f"Error saving image: {str(e)}", "#FF5722")

    def load_image(self, file_path):
        """Load image from file path"""
        try:
            image = Image.open(file_path)
            # Resize image to fit canvas while maintaining aspect ratio
            display_size = (450, 450)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)

            self.input_image = image
            self.display_image(image, self.input_canvas)
            
            # Clear output
            self.output_image = None
            self.output_canvas.delete("all")
            self.output_canvas.create_text(
                225, 225,
                text="Processed image will appear here",
                fill='#666666',
                font=('Arial', 14),
                justify=tk.CENTER
            )
            
            self.image_loaded = True
            
            # Enable apply button if algorithms are selected
            if self.algorithm_sequence:
                self.apply_btn.configure(state="normal")
                
            self.show_message(f"Image loaded: {os.path.basename(file_path)}", "#4CAF50")
            
        except Exception as e:
            self.image_loaded = False
            self.apply_btn.configure(state="disabled")
            self.show_message(f"Error loading image: {str(e)}", "#FF5722")

    @staticmethod
    def display_image(image: Image.Image, canvas: tk.Canvas):
        """Display image on canvas"""
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

    def show_message(self, message: str, color: str = "#2196F3", duration: int = 3000):
        """Show a temporary message to the user."""
        # Cancel any existing message timer
        if self.message_after_id:
            self.app.after_cancel(self.message_after_id)

        # Show new message
        self.message_label.configure(text=message, text_color=color)

        # Schedule message removal
        self.message_after_id = self.app.after(duration, lambda: self.message_label.configure(text="Ready", text_color="#4CAF50"))

    def start_processing(self):
        """Start image processing in a separate thread"""
        if self.is_processing:
            return

        if not self.input_image or not self.algorithm_sequence:
            self.show_message("Please load an image and add algorithms to the sequence", "#FF9800")
            return

        # Disable buttons during processing
        self.apply_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")

        # Show progress bar
        self.progress_bar.set(0)
        self.show_progress()
        self.status_label.configure(text="Starting processing...")
        self.is_processing = True

        # Start processing thread
        thread = threading.Thread(target=self.process_image_thread)
        thread.daemon = True
        thread.start()

        # Start checking progress
        self.app.after(100, self.check_processing)

    def process_image_thread(self):
        """Process image through algorithm sequence in a separate thread"""
        try:
            processed_image = self.input_image.copy()
            total_steps = len(self.algorithm_sequence)

            # Track if we're in frequency domain (only for traditional algorithms)
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

                    # Handle neural network algorithms differently
                    if self.current_mode == 'neural':
                        # Neural network algorithms return processed images directly
                        if alg["name"] in ["FeatureMatching", "FeatureMatchingHomography"]:
                            # These algorithms need two images - for now, use the same image
                            result = alg["function"](processed_image, processed_image, **alg.get("params", {}))
                            if isinstance(result, tuple):
                                processed_image = result[0]  # Get the matched image
                            else:
                                processed_image = result
                        else:
                            # Other neural algorithms
                            result = alg["function"](processed_image, **alg.get("params", {}))
                            if isinstance(result, tuple):
                                processed_image = result[0]  # Get the processed image
                            else:
                                processed_image = result
                    else:
                        # Traditional image processing algorithms
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

                    # Convert back to PIL Image if not in frequency domain (for traditional) or always for neural
                    if (self.current_mode == 'neural' or not in_frequency_domain) and isinstance(processed_image, np.ndarray):
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
        """Update progress bar and status"""
        self.progress_bar.set(progress)
        self.status_label.configure(text=status)

    def show_error(self, error_message):
        """Show error message and stop processing"""
        self.show_message(error_message, "#FF5722")
        self.hide_progress()
        self.enable_buttons()

    def check_processing(self):
        """Check if processing is complete"""
        if self.is_processing:
            self.app.after(100, self.check_processing)
        else:
            self.enable_buttons()
            self.app.after(2000, self.hide_progress)

    def enable_buttons(self):
        """Enable all buttons after processing"""
        self.apply_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        self.open_btn.configure(state="normal")
        self.save_btn.configure(state="normal")

    def get_algorithm_module(self, name: str):
        """Get algorithm module by name - used by medical enhancement GUI"""
        try:
            module = importlib.import_module(f"algorithm_image_processing.{name}")
            return getattr(module, name)
        except (ImportError, AttributeError):
            return None

    def run(self):
        """Start the application"""
        self.app.mainloop()


if __name__ == '__main__':
    app = ImageProcessorApp()
    app.run()