import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
from typing import Dict, Any, Optional, Final
import numpy as np

# Medical image enhancement sequences
MEDICAL_SEQUENCES: Final = {
    "X-ray": [
        {"name": "Rgb2Gray", "params": {}},
        {"name": "ContrastStretching", "params": {"low_percentile": 1, "high_percentile": 99}},  # More aggressive contrast
        {"name": "GaussianLowPassFilter", "params": {"sigma": 20}},  # Reduce noise while preserving edges
        {"name": "HistogramEqualization", "params": {}},  # Enhance overall contrast
        {"name": "PointSharpening", "params": {"factor": 1.2}}  # Subtle edge enhancement
    ],
    "MRI": [
        {"name": "Rgb2Gray", "params": {}},
        {"name": "ContrastStretching", "params": {"low_percentile": 2, "high_percentile": 98}},
        {"name": "MedianFilter", "params": {"size": 3}},  # Remove speckle noise
        {"name": "GaussianHighPassFilter", "params": {"sigma": 40}},  # Enhance tissue boundaries
        {"name": "HistogramEqualization", "params": {}}  # Final contrast adjustment
    ],
    "CT Scan": [
        {"name": "Rgb2Gray", "params": {}},
        {"name": "ContrastStretching", "params": {"low_percentile": 1, "high_percentile": 99}},
        {"name": "GaussianLowPassFilter", "params": {"sigma": 15}},  # Preserve fine details
        {"name": "HistogramEqualization", "params": {}},
        {"name": "PointSharpening", "params": {"factor": 1.3}}  # Moderate sharpening
    ],
    "Ultrasound": [
        {"name": "Rgb2Gray", "params": {}},
        {"name": "MedianFilter", "params": {"size": 5}},  # Stronger speckle noise removal
        {"name": "ContrastStretching", "params": {"low_percentile": 2, "high_percentile": 98}},
        {"name": "GaussianLowPassFilter", "params": {"sigma": 25}},  # Smooth while preserving boundaries
        {"name": "HistogramEqualization", "params": {}}
    ],
    "Bone Density": [
        {"name": "Rgb2Gray", "params": {}},
        {"name": "ContrastStretching", "params": {"low_percentile": 1, "high_percentile": 99}},
        {"name": "GaussianHighPassFilter", "params": {"sigma": 35}},  # Strong edge enhancement
        {"name": "HistogramEqualization", "params": {}},
        {"name": "PointSharpening", "params": {"factor": 1.5}},  # Strong sharpening
        {"name": "MedianFilter", "params": {"size": 3}}  # Final noise cleanup
    ],
    "Blood Vessel": [
        {"name": "Rgb2Gray", "params": {}},
        {"name": "ContrastStretching", "params": {"low_percentile": 1, "high_percentile": 99}},
        {"name": "MedianFilter", "params": {"size": 3}},  # Initial noise removal
        {"name": "GaussianHighPassFilter", "params": {"sigma": 30}},  # Enhance vessel edges
        {"name": "HistogramEqualization", "params": {}},
        {"name": "PointSharpening", "params": {"factor": 1.4}}  # Enhance vessel boundaries
    ]
}

class MedicalEnhancementGUI:
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.window = None
        self.current_sequence = None
        self.preview_image = None
        self.original_image = None
        self.is_processing = False
        
    def show(self):
        if self.window is None or not self.window.winfo_exists():
            self.create_window()
        else:
            self.window.lift()
            
    def create_window(self):
        self.window = ctk.CTkToplevel()
        self.window.title("Medical Image Enhancement")
        self.window.geometry("1200x800")
        
        # Create main frames
        left_frame = ctk.CTkFrame(self.window, width=300)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        right_frame = ctk.CTkFrame(self.window)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Medical sequence selection
        sequence_frame = ctk.CTkFrame(left_frame)
        sequence_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(sequence_frame, text="Enhancement Type:").pack(pady=5)
        
        # Dropdown for medical sequence types
        self.sequence_var = tk.StringVar(value="X-ray")
        sequence_dropdown = ctk.CTkOptionMenu(
            sequence_frame,
            values=list(MEDICAL_SEQUENCES.keys()),
            variable=self.sequence_var,
            command=self.on_sequence_change
        )
        sequence_dropdown.pack(pady=5)
        
        # Preview frame
        preview_frame = ctk.CTkFrame(right_frame)
        preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.preview_canvas = tk.Canvas(
            preview_frame,
            bg='#333333',
            highlightthickness=0
        )
        self.preview_canvas.pack(fill="both", expand=True)
        
        # Control buttons
        button_frame = ctk.CTkFrame(left_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self.apply_btn = ctk.CTkButton(
            button_frame,
            text="Apply Enhancement",
            command=self.apply_enhancement,
            state="disabled"  # Initially disabled
        )
        self.apply_btn.pack(fill="x", pady=5)
        
        self.reset_btn = ctk.CTkButton(
            button_frame,
            text="Reset",
            command=self.reset_preview,
            state="disabled"  # Initially disabled
        )
        self.reset_btn.pack(fill="x", pady=5)
        
        # Status label
        self.status_label = ctk.CTkLabel(left_frame, text="")
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(left_frame)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        self.progress_bar.set(0)
        
        # Sequence preview
        sequence_list_frame = ctk.CTkFrame(left_frame)
        sequence_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(sequence_list_frame, text="Enhancement Steps:").pack(pady=5)
        
        self.sequence_text = ctk.CTkTextbox(sequence_list_frame, height=200)
        self.sequence_text.pack(fill="both", expand=True)
        
        self.update_sequence_preview()
        
    def on_sequence_change(self, selection):
        if self.is_processing:
            return
            
        self.current_sequence = MEDICAL_SEQUENCES[selection]
        self.update_sequence_preview()
        
        # Check if input_image exists and is not None
        if hasattr(self.parent_app, 'input_image') and self.parent_app.input_image is not None:
            self.preview_enhancement()
            
    def update_sequence_preview(self):
        if not self.current_sequence:
            self.current_sequence = MEDICAL_SEQUENCES["X-ray"]
            
        self.sequence_text.delete("1.0", tk.END)
        for idx, step in enumerate(self.current_sequence, 1):
            self.sequence_text.insert(tk.END, f"{idx}. {step['name']}\n")
            if step['params']:
                for param, value in step['params'].items():
                    self.sequence_text.insert(tk.END, f"   - {param}: {value}\n")
                    
    def preview_enhancement(self):
        if self.is_processing:
            return
            
        self.is_processing = True
        self.progress_bar.set(0)
        
        # Check if input_image exists and is not None
        if not hasattr(self.parent_app, 'input_image') or self.parent_app.input_image is None:
            self.status_label.configure(text="No image loaded!")
            self.is_processing = False
            return
            
        try:
            # Convert NumPy array to PIL Image if needed
            if isinstance(self.parent_app.input_image, np.ndarray):
                input_img = Image.fromarray(self.parent_app.input_image.astype('uint8'))
            else:
                input_img = self.parent_app.input_image
                
            self.original_image = input_img.copy()
            enhanced_image = input_img.copy()
            
            total_steps = len(self.current_sequence)
            for idx, step in enumerate(self.current_sequence):
                progress = (idx + 1) / total_steps
                self.progress_bar.set(progress)
                self.status_label.configure(text=f"Applying {step['name']}...")
                
                # Get algorithm class
                alg_module = self.parent_app.get_algorithm_module(step['name'])
                if alg_module:
                    enhanced_image = alg_module(enhanced_image, **step['params'])
                    
                self.window.update()
                
            # Convert back to PIL Image if needed
            if isinstance(enhanced_image, np.ndarray):
                enhanced_image = Image.fromarray(enhanced_image.astype('uint8'))
                
            self.preview_image = enhanced_image
            self.show_preview()
            self.status_label.configure(text="Enhancement preview ready!")
            self.apply_btn.configure(state="normal")
            self.reset_btn.configure(state="normal")
            
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")
            
        finally:
            self.progress_bar.set(1)
            self.is_processing = False
            
    def show_preview(self):
        if self.preview_image is None:
            return
            
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width == 0 or canvas_height == 0:
            # Canvas not ready yet, try again after a short delay
            self.window.after(100, self.show_preview)
            return
            
        img_width, img_height = self.preview_image.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = self.preview_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        
        # Center image on canvas
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(x, y, anchor="nw", image=self.tk_image)
        
    def apply_enhancement(self):
        if self.preview_image is None:
            return
            
        try:
            # Convert to PIL Image if needed
            if isinstance(self.preview_image, np.ndarray):
                preview_img = Image.fromarray(self.preview_image.astype('uint8'))
            else:
                preview_img = self.preview_image
                
            # Update both input and output images in main window
            self.parent_app.input_image = preview_img.copy()
            self.parent_app.current_image = preview_img.copy()
            
            # Update the display in main window
            if hasattr(self.parent_app, 'display_image'):
                self.parent_app.display_image(preview_img, self.parent_app.input_canvas)
            if hasattr(self.parent_app, 'update_output_image'):
                self.parent_app.update_output_image()
                
            self.status_label.configure(text="Enhancement applied to main window!")
            
        except Exception as e:
            self.status_label.configure(text=f"Error applying enhancement: {str(e)}")
            
    def reset_preview(self):
        if self.original_image is None:
            return
            
        self.preview_image = self.original_image.copy()
        self.show_preview()
        self.status_label.configure(text="Reset to original image")
        self.progress_bar.set(0)
        self.apply_btn.configure(state="disabled")
