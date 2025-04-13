import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons, TextBox
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
# Import patheffects explicitly to avoid attribute error
from matplotlib import patheffects
import cv2
import os
import sys
from glob import glob
from datetime import datetime
import shutil
import json
from collections import Counter, defaultdict
import pandas as pd

# Set style for a professional look - use a more compatible approach
plt.rcParams['figure.facecolor'] = '#1F1F1F'
plt.rcParams['axes.facecolor'] = '#2F2F2F'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#3F3F3F'

# Try to import seaborn for enhanced styling but handle if not available
try:
    import seaborn as sns
    sns.set_style("darkgrid")
    has_seaborn = True
except ImportError:
    has_seaborn = False

class MultiMaskViewer:
    def __init__(self, base_folder, classes_csv=None):
        """
        Initialize the viewer with the base folder containing images and labels
        Supports multiple object masks per image
        
        Args:
            base_folder: Path to folder containing images and labels subfolders
            classes_csv: Optional path to CSV file with class names
        """
        # Basic folder setup
        self.base_folder = os.path.normpath(base_folder)
        self.images_folder = os.path.join(self.base_folder, 'images')
        self.labels_folder = os.path.join(self.base_folder, 'labels')
        
        # Create output directories
        self.output_folder = os.path.join(self.base_folder, 'saved_visualizations')
        self.faulty_folder = os.path.join(self.base_folder, 'faulty_folder')
        self.faulty_images = os.path.join(self.faulty_folder, 'faulty_images')
        self.faulty_labels = os.path.join(self.faulty_folder, 'faulty_labels')
        
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.faulty_images, exist_ok=True)
        os.makedirs(self.faulty_labels, exist_ok=True)
        
        # Get list of all txt files and sort them
        self.txt_files = sorted(glob(os.path.join(self.labels_folder, '*.txt')))
        if not self.txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.labels_folder}")
        
        # Load class names if available
        self.class_names = {}
        if classes_csv:
            try:
                df = pd.read_csv(classes_csv)
                if 'class_name' in df.columns:
                    # Create a dictionary mapping class IDs to class names
                    for i, name in enumerate(df['class_name']):
                        self.class_names[i] = name
                    print(f"Loaded {len(self.class_names)} class names from {classes_csv}")
            except Exception as e:
                print(f"Warning: Could not load class names from {classes_csv}: {e}")
        
        # Create class-to-color mapping
        self.class_colors = [
            (0, 0.8, 0),        # Green
            (0.8, 0, 0),        # Red
            (0, 0, 0.8),        # Blue
            (0.8, 0.8, 0),      # Yellow
            (0.8, 0, 0.8),      # Magenta
            (0, 0.8, 0.8),      # Cyan
            (1.0, 0.4, 0),      # Orange
            (0.4, 0, 1.0),      # Purple
            (0.4, 0.4, 0.4),    # Gray
            (0.8, 0.4, 0.2),    # Brown
        ]
        
        # View state
        self.current_index = 0
        self.image_size = (800, 600)
        self.mask_alpha = 0.5
        self.show_contours = True
        self.display_mode = "overlay"  # Can be "overlay", "side_by_side", or "masked"
        self.show_points = True
        self.show_class_ids = True
        self.highlight_object = None
        
        # Statistics tracking
        self.dataset_stats = {
            "total_images": len(self.txt_files),
            "total_annotations": 0,
            "class_distribution": Counter(),
            "objects_per_image": [],
            "avg_vertices_per_mask": 0,
            "faulty_count": len(os.listdir(self.faulty_images)) if os.path.exists(self.faulty_images) else 0
        }
        
        # Calculate initial statistics
        self.calculate_dataset_stats()
        
        # Initialize the UI
        self.setup_ui()
        
        # Display first image
        self.update_display()
        
    def setup_ui(self):
        """Set up a simple UI layout with fixed positions for reliable display"""
        # Create a figure with a specific size and no constrained_layout (can cause issues with widgets)
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title('SAM Annotation Visualizer')
        
        # Define fixed regions for each component with explicit coordinates
        # Format is [left, bottom, width, height] in figure coordinates (0-1)
        
        # Main regions
        main_display_pos = [0.05, 0.38, 0.65, 0.55]  # Main image
        controls_pos = [0.75, 0.38, 0.2, 0.55]       # Controls/options panel (replacing stats)
        
        # Detailed views row
        details_height = 0.25
        details_y = 0.08
        details_width = 0.28
        original_pos = [0.05, details_y, details_width, details_height]
        mask_pos = [0.36, details_y, details_width, details_height]
        overlay_pos = [0.67, details_y, details_width, details_height]
        
        # Create axes for all display areas
        self.main_display_ax = self.fig.add_axes(main_display_pos)
        self.controls_panel = self.fig.add_axes(controls_pos)
        self.original_ax = self.fig.add_axes(original_pos)
        self.mask_ax = self.fig.add_axes(mask_pos)
        self.overlay_ax = self.fig.add_axes(overlay_pos)
        
        # Set titles with consistent positioning using the dedicated method
        self.set_section_titles()
        
        # Turn off axes
        for ax in [self.main_display_ax, self.controls_panel, self.original_ax, 
                  self.mask_ax, self.overlay_ax]:
            ax.axis('off')
            
        # Add controls directly to figure
        self.setup_controls()

    def setup_controls(self):
        """Set up controls using fixed coordinates for reliable positioning"""
        # Fixed positions for buttons in figure coordinates (0-1)
        button_height = 0.05
        button_width = 0.1
        button_y = 0.01  # Bottom row
        spacing = 0.01
        
        # Navigation buttons (left side)
        prev_btn_pos = [0.05, button_y, button_width, button_height]
        next_btn_pos = [0.05 + button_width + spacing, button_y, button_width, button_height]
        
        # Action buttons (center)
        save_btn_pos = [0.35, button_y, button_width, button_height]
        faulty_btn_pos = [0.35 + button_width + spacing, button_y, button_width, button_height]
        
        # Opacity slider (right)
        slider_pos = [0.65, button_y, 0.3, button_height]
        
        # Create navigation buttons
        self.prev_button_ax = self.fig.add_axes(prev_btn_pos)
        self.next_button_ax = self.fig.add_axes(next_btn_pos)
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.next_button = Button(self.next_button_ax, 'Next')
        
        # Set button colors
        self.prev_button.color = '#2c3e50'
        self.prev_button.hovercolor = '#34495e'
        self.next_button.color = '#2c3e50'
        self.next_button.hovercolor = '#34495e'
        
        # Create action buttons
        self.save_button_ax = self.fig.add_axes(save_btn_pos)
        self.faulty_button_ax = self.fig.add_axes(faulty_btn_pos)
        self.save_button = Button(self.save_button_ax, 'Save')
        self.faulty_button = Button(self.faulty_button_ax, 'Mark Faulty')
        
        # Set button colors
        self.save_button.color = '#2980b9'
        self.save_button.hovercolor = '#3498db'
        self.faulty_button.color = '#c0392b'
        self.faulty_button.hovercolor = '#e74c3c'
        
        # Create opacity slider
        self.alpha_slider_ax = self.fig.add_axes(slider_pos)
        self.alpha_slider = Slider(
            self.alpha_slider_ax, 'Mask Opacity', 0.0, 1.0, 
            valinit=self.mask_alpha
        )
        self.alpha_slider.poly.set_facecolor('#8e44ad')  # Slider color
        
        # Get the position of the controls panel
        cp_pos = self.controls_panel.get_position()
        
        # Arrange controls in the controls panel with more space between them
        
        # First row: Checkboxes directly under the title with more gap
        check_height = 0.1
        check_width = 0.18
        check_y = cp_pos.y0 + cp_pos.height - 0.15  # Position under the title
        check_pos = [cp_pos.x0 + 0.01, check_y, check_width, check_height]
        
        # Create display options checkboxes
        self.options_ax = self.fig.add_axes(check_pos)
        self.options_checkbox = CheckButtons(
            self.options_ax, ['Show Contours', 'Show Points', 'Show Class IDs'],
            [self.show_contours, self.show_points, self.show_class_ids]
        )
        
        # Try to set colors for compatibility
        try:
            for rect in self.options_checkbox.rectangles:
                rect.set_facecolor('#27ae60')
        except AttributeError:
            pass
        
        # Second row: Radio buttons - positioned with minimal gap to checkboxes
        radio_height = 0.1
        radio_width = 0.18
        radio_y = check_y - radio_height - 0.02  # Reduce gap to almost touching (was 0.04)
        radio_pos = [cp_pos.x0 + 0.01, radio_y, radio_width, radio_height]
        
        # Create display mode radio buttons
        self.display_mode_ax = self.fig.add_axes(radio_pos)
        self.display_mode_selector = RadioButtons(
            self.display_mode_ax, ['Overlay', 'Side by Side', 'Masked'],
            active=0
        )
        
        # Try to set the active color for compatibility
        try:
            for circle in self.display_mode_selector.circles:
                circle.set_facecolor('#3498db')
        except AttributeError:
            pass
        
        # Connect events
        self.prev_button.on_clicked(self.previous_image)
        self.next_button.on_clicked(self.next_image)
        self.save_button.on_clicked(self.save_figure)
        self.faulty_button.on_clicked(self.mark_as_faulty)
        self.alpha_slider.on_changed(self.update_alpha)
        self.display_mode_selector.on_clicked(self.change_display_mode)
        self.options_checkbox.on_clicked(self.toggle_option)
        
        # Add keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def calculate_dataset_stats(self):
        """Calculate dataset statistics for visualization"""
        total_vertices = 0
        total_objects = 0
        objects_per_image = []
        
        # Process each annotation file
        for txt_file in self.txt_files:
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                
                num_objects = len(lines)
                objects_per_image.append(num_objects)
                total_objects += num_objects
                
                # Process each line as a separate object
                for line in lines:
                    coordinates = list(map(float, line.strip().split()))
                    class_id = int(coordinates[0])  # First number is class ID
                    num_vertices = (len(coordinates) - 1) // 2  # Number of (x,y) vertices
                    
                    # Update class distribution
                    self.dataset_stats["class_distribution"][class_id] += 1
                    
                    # Add vertices count
                    total_vertices += num_vertices
            except Exception as e:
                print(f"Error processing {txt_file}: {str(e)}")
                continue
        
        # Update statistics
        self.dataset_stats["total_annotations"] = total_objects
        self.dataset_stats["objects_per_image"] = objects_per_image
        
        if total_objects > 0:
            self.dataset_stats["avg_vertices_per_mask"] = total_vertices / total_objects

    def load_and_process_image(self, txt_path):
        """Load and process a single image with multiple object masks"""
        # Generate corresponding image path using OS-independent path handling
        txt_filename = os.path.basename(txt_path)
        img_filename = os.path.splitext(txt_filename)[0] + '.png'
        img_path = os.path.join(self.images_folder, img_filename)
        
        if not os.path.exists(img_path):
            # Try other common image extensions if png not found
            for ext in ['.jpg', '.jpeg', '.tif', '.tiff']:
                alt_img_path = os.path.join(self.images_folder, os.path.splitext(txt_filename)[0] + ext)
                if os.path.exists(alt_img_path):
                    img_path = alt_img_path
                    break
            else:
                raise FileNotFoundError(f"Image not found: {img_path} (tried common extensions)")
        
        # Read and resize the original image
        original_img = cv2.imread(img_path)
        if original_img is None:
            raise ValueError(f"Could not read image: {img_path}")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, self.image_size)
        
        # Create mask image (single channel for combined mask)
        combined_mask = np.zeros(self.image_size[::-1], dtype=np.uint8)
        
        # Read multiple object coordinates from file
        individual_masks = []
        object_data = []
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            # Process each line as a separate object
            for line in lines:
                coordinates = list(map(float, line.strip().split()))
                class_id = int(coordinates[0])  # First number is class ID
                
                # Convert normalized coordinates to pixel coordinates for this object
                points = []
                for i in range(1, len(coordinates), 2):
                    x = int(coordinates[i] * self.image_size[0])
                    y = int(coordinates[i + 1] * self.image_size[1])
                    points.append([x, y])
                
                # Create individual mask for this object
                obj_mask = np.zeros(self.image_size[::-1], dtype=np.uint8)
                points = np.array(points, np.int32)
                cv2.fillPoly(obj_mask, [points], 255)
                
                # Calculate object properties
                area = cv2.countNonZero(obj_mask)
                perimeter = cv2.arcLength(points, True)
                x, y, w, h = cv2.boundingRect(points)
                
                # Store object data
                object_data.append({
                    'class_id': class_id,
                    'points': points,
                    'area': area,
                    'perimeter': perimeter,
                    'bounding_box': (x, y, w, h),
                    'num_vertices': len(points)
                })
                
                individual_masks.append(obj_mask)
                
                # Add to combined mask
                cv2.fillPoly(combined_mask, [points], 255)
                
        except Exception as e:
            print(f"Error processing annotation {txt_path}: {str(e)}")
            # Return a blank visualization with error message
            return original_img, np.zeros(self.image_size[::-1], dtype=np.uint8), original_img, os.path.basename(txt_path), []
        
        # Create overlay with different colors for each object
        overlay_img = original_img.copy()
        
        for idx, (obj_mask, obj_info) in enumerate(zip(individual_masks, object_data)):
            # Get color for this class
            class_id = obj_info['class_id']
            rgb_color = self.class_colors[class_id % len(self.class_colors)]
            
            # Create color mask
            color_mask = np.zeros_like(original_img)
            color = tuple(int(c * 255) for c in rgb_color)
            color_mask[obj_mask > 0] = color
            
            # Apply mask with current opacity
            overlay_img = cv2.addWeighted(overlay_img, 1, color_mask, self.mask_alpha, 0)
            
            # Draw contours if enabled
            if self.show_contours:
                contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_img, contours, -1, color, 2)
            
            # Draw class ID if enabled
            if self.show_class_ids:
                x, y, w, h = obj_info['bounding_box']
                class_id = obj_info['class_id']
                # Use class name if available, otherwise just show class ID
                if class_id in self.class_names:
                    class_text = f"{self.class_names[class_id]} (Class {class_id})"
                else:
                    class_text = f"Class {class_id}"
                
                cv2.putText(
                    overlay_img, class_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
        
        return original_img, combined_mask, overlay_img, os.path.basename(txt_path), object_data

    def set_section_titles(self):
        """Set section titles with consistent positioning"""
        # Clear existing text annotations first
        for txt in self.fig.texts[:]:
            if hasattr(self.fig, '_suptitle') and txt is self.fig._suptitle:
                continue
            txt.remove()
        
        # Get positions from axes for precise positioning
        main_pos = self.main_display_ax.get_position()
        controls_pos = self.controls_panel.get_position()
        original_pos = self.original_ax.get_position()
        mask_pos = self.mask_ax.get_position()
        overlay_pos = self.overlay_ax.get_position()
        
        # Set titles with enhanced styling
        title_props = dict(fontsize=14, fontweight='bold', color='white',
                          bbox=dict(facecolor='#1E5C85', edgecolor='#6BBBFF', pad=5.0, 
                                   boxstyle='round,pad=0.5', alpha=0.9))
        
        # Replace "Annotation Visualization" with image counter information
        if self.txt_files:
            filename = os.path.basename(self.txt_files[self.current_index])
            total_images = len(self.txt_files)
            
            # Use the title_props styling for the image counter
            self.fig.text(main_pos.x0 + main_pos.width/2, 
                         main_pos.y0 + main_pos.height + 0.01, 
                         f"Image {self.current_index + 1}/{total_images} - {filename}", 
                         ha='center', va='center', **title_props)
        else:
            # Fallback if no files
            self.fig.text(main_pos.x0 + main_pos.width/2, 
                         main_pos.y0 + main_pos.height + 0.01, 
                         "No Images Available", 
                         ha='center', va='center', **title_props)
        
        self.fig.text(controls_pos.x0 + controls_pos.width/2, 
                     controls_pos.y0 + controls_pos.height + 0.01, 
                     'Display Options', 
                     ha='center', va='center', **title_props)
        
        # Slightly smaller style for the detail panels
        detail_title_props = dict(fontsize=12, fontweight='bold', color='white',
                                 bbox=dict(facecolor='#444F52', edgecolor='#9DDCFF', pad=3.0, 
                                          boxstyle='round,pad=0.3', alpha=0.9))
        
        self.fig.text(original_pos.x0 + original_pos.width/2, 
                     original_pos.y0 + original_pos.height + 0.02, 
                     'Original Image', 
                     ha='center', va='center', **detail_title_props)
        
        self.fig.text(mask_pos.x0 + mask_pos.width/2, 
                     mask_pos.y0 + mask_pos.height + 0.02, 
                     'Segmentation Mask', 
                     ha='center', va='center', **detail_title_props)
        
        self.fig.text(overlay_pos.x0 + overlay_pos.width/2, 
                     overlay_pos.y0 + overlay_pos.height + 0.02, 
                     'Colored Objects', 
                     ha='center', va='center', **detail_title_props)

    def update_display(self):
        """Update the display with current image"""
        if not self.txt_files:
            plt.suptitle("No images available")
            return
            
        try:
            # Clear all axes
            self.main_display_ax.clear()
            self.original_ax.clear()
            self.mask_ax.clear()
            self.overlay_ax.clear()
            self.controls_panel.clear()
            
            # Set section titles with consistent positioning
            self.set_section_titles()
            
            # Turn off axes
            for ax in [self.main_display_ax, self.controls_panel, self.original_ax, self.mask_ax, self.overlay_ax]:
                ax.axis('off')
            
            # Load and process current image
            try:
                original, mask, overlay, filename, object_data = self.load_and_process_image(
                    self.txt_files[self.current_index]
                )
            except (FileNotFoundError, ValueError) as e:
                # Handle file-related errors gracefully
                error_msg = str(e)
                print(f"Error during display update: {error_msg}")
                
                # Create blank images
                original = np.zeros(self.image_size[::-1] + (3,), dtype=np.uint8)
                mask = np.zeros(self.image_size[::-1], dtype=np.uint8)
                overlay = original.copy()
                filename = os.path.basename(self.txt_files[self.current_index])
                object_data = []
                
                # Display error message on main display
                self.main_display_ax.text(0.5, 0.5, f"Error: {error_msg}", 
                               ha='center', va='center', color='red', fontsize=14,
                               transform=self.main_display_ax.transAxes)
            
            # Show images in the smaller panels
            self.original_ax.imshow(original)
            self.mask_ax.imshow(mask, cmap='gray')
            self.overlay_ax.imshow(overlay)
            
            # Show main display based on current mode
            if object_data:  # Only show display mode if we have valid data
                if self.display_mode == "overlay":
                    self.main_display_ax.imshow(overlay)
                elif self.display_mode == "side_by_side":
                    # Fix side-by-side composite - ensure mask is properly converted to RGB
                    # Create a proper RGB representation of the mask for side-by-side view
                    mask_rgb = np.zeros_like(original)
                    mask_rgb[:,:,0] = mask  # Set all RGB channels to the mask values
                    mask_rgb[:,:,1] = mask
                    mask_rgb[:,:,2] = mask
                    
                    # Create side-by-side composite with proper dimensions
                    h, w, _ = original.shape
                    composite = np.zeros((h, w*2, 3), dtype=np.uint8)
                    composite[:, :w, :] = original
                    composite[:, w:, :] = mask_rgb
                    
                    self.main_display_ax.imshow(composite)
                elif self.display_mode == "masked":
                    # Create masked original
                    masked = original.copy()
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    masked = masked * mask_rgb
                    self.main_display_ax.imshow(masked)
            
            # Add class distribution to controls panel
            self.update_class_distribution()
            
            # Draw everything
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error during display update: {str(e)}")
            import traceback
            traceback.print_exc()  # Print detailed error information
            
    def update_class_distribution(self):
        """Update only the class distribution visualization in the controls panel"""
        stats = self.dataset_stats
        
        # Only show class distribution pie chart
        if stats['class_distribution']:
            # Create labels with class names if available
            labels = []
            for k in stats['class_distribution'].keys():
                if k in self.class_names:
                    labels.append(f"{self.class_names[k]} (Class {k})")
                else:
                    labels.append(f"Class {k}")
                    
            sizes = list(stats['class_distribution'].values())
            colors = [self.class_colors[k % len(self.class_colors)] for k in stats['class_distribution'].keys()]
            
            try:
                # Get the position of the controls panel
                cp_pos = self.controls_panel.get_position()
                
                # Calculate a safe position for the pie chart below radio buttons
                # First get the position of radio buttons to calculate safe distance
                if hasattr(self, 'display_mode_ax'):
                    radio_pos = self.display_mode_ax.get_position()
                    # Move everything significantly farther down by increasing this gap value from 0.18 to 0.28
                    pie_top_y = max(0.02, radio_pos.y0 - cp_pos.y0 - 0.28)
                else:
                    # Default if radio buttons not found
                    pie_top_y = 0.2
                
                # Keep the pie height the same
                pie_height = 0.45
                
                # Clear any existing pie chart
                for ax in self.fig.axes[:]:
                    if ax not in [self.main_display_ax, self.controls_panel, self.original_ax,
                                  self.mask_ax, self.overlay_ax, self.options_ax, 
                                  self.display_mode_ax, self.prev_button_ax, self.next_button_ax,
                                  self.save_button_ax, self.faulty_button_ax, self.alpha_slider_ax]:
                        ax.remove()
                
                # Create a separate axis just for the title - position it right on top of the pie chart
                title_height = 0.05  # Keep increased height for title
                # Title position moved to be directly on top of pie chart (change from +0.02 to -0.03)
                title_ax = self.controls_panel.inset_axes([0.1, pie_top_y + pie_height - 0.08, 0.8, title_height])
                title_ax.axis('off')  # Make sure axis is invisible
                
                # Make title more prominent but keep it close to the pie chart
                title_ax.set_title('Class Distribution', 
                                 fontsize=14,  # Keep increased font size 
                                 fontweight='bold', color='white',
                                 bbox=dict(facecolor='#1E5C85', edgecolor='#6BBBFF', 
                                         boxstyle='round,pad=0.3', alpha=0.9))  # Keep more padding
                
                # Make sure title is visible by bringing it to front
                title_ax.set_zorder(100)
                
                # Now create the pie chart axis below it - which is now lower in the panel
                pie_ax = self.controls_panel.inset_axes([0.0002, pie_top_y, 0.85, pie_height])
                
                # Make the pie chart larger by increasing the radius
                wedges, texts, autotexts = pie_ax.pie(
                    sizes, 
                    colors=colors, 
                    autopct='%1.1f%%', 
                    pctdistance=0.8,
                    textprops=dict(fontsize=12, fontweight='bold'),
                    startangle=90,
                    radius=0.95,  # Increased to make pie chart bigger (was 0.85)
                    wedgeprops=dict(width=0.7, edgecolor='w')
                )
                
                # Set pie chart text color to be more visible
                for autotext in autotexts:
                    autotext.set_color('white')
                    # Add a subtle shadow effect using correctly imported patheffects
                    try:
                        autotext.set_path_effects([
                            patheffects.withStroke(linewidth=3, foreground='black')
                        ])
                    except Exception as e:
                        # Fallback if patheffects don't work
                        print(f"Notice: Could not apply path effects: {e}")
                
                # Add legend separately with explicit positioning to ensure it's visible
                if len(labels) <= 20:  # Show legend for more classes
                    try:
                        # Create legend patches manually to ensure correct ordering
                        from matplotlib.patches import Patch
                        legend_elements = [Patch(facecolor=colors[i], edgecolor='w', label=labels[i]) 
                                         for i in range(len(labels))]
                        
                        # Position legend to the right of the pie chart with tighter parameters
                        legend = pie_ax.legend(
                            handles=legend_elements, 
                            fontsize=8,  # Smaller font for more compact display
                            loc='center left',  # Position legend centered on left side
                            bbox_to_anchor=(0.9, 0.5),  # Position legend to right of pie chart
                            frameon=True,
                            facecolor='#2F2F2F',
                            edgecolor='white',
                            # Make legend more compact
                            borderaxespad=0.05,  # Reduced from 0.1
                            handletextpad=0.3,  # Reduced from 0.5
                            columnspacing=0.8,  # Reduced from 1.0
                            labelspacing=0.4   # Reduced space between legend items
                        )
                        
                        # Force legend to be drawn
                        legend.set_visible(True)
                        
                        # Ensure legend text is visible
                        for text in legend.get_texts():
                            text.set_color('white')
                            text.set_fontweight('bold')  # Make text bold
                            
                        # Force redraw
                        pie_ax.figure.canvas.draw_idle()
                    except Exception as e:
                        print(f"Error with legend: {str(e)}")
            except Exception as e:
                print(f"Warning: Could not render pie chart: {str(e)}")

    def update_alpha(self, val):
        """Update mask opacity when slider is changed"""
        self.mask_alpha = val
        self.update_display()
    
    def change_display_mode(self, label):
        """Change the display mode"""
        if label == 'Overlay':
            self.display_mode = "overlay"
        elif label == 'Side by Side':
            self.display_mode = "side_by_side"
        elif label == 'Masked':
            self.display_mode = "masked"
        self.update_display()
    
    def toggle_option(self, label):
        """Toggle display options"""
        if label == 'Show Contours':
            self.show_contours = not self.show_contours
        elif label == 'Show Points':
            self.show_points = not self.show_points
        elif label == 'Show Class IDs':
            self.show_class_ids = not self.show_class_ids
        self.update_display()

    def save_figure(self, event=None):
        """Save visualization of current image with all objects"""
        if not self.txt_files:
            return
            
        try:
            # Get current file information
            current_filename = os.path.splitext(os.path.basename(self.txt_files[self.current_index]))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"{current_filename}_visualization_{timestamp}.png"
            save_path = os.path.join(self.output_folder, save_filename)
            
            # Create figure with fixed layout
            save_fig = plt.figure(figsize=(16, 10), dpi=150)
            save_fig.canvas.manager.set_window_title(f'Visualization - {current_filename}')
            
            # Define fixed positions for regions
            main_pos = [0.05, 0.35, 0.9, 0.6]
            original_pos = [0.1, 0.05, 0.35, 0.25]
            mask_pos = [0.55, 0.05, 0.35, 0.25]
            
            # Create axes directly 
            main_ax = save_fig.add_axes(main_pos)
            main_ax.set_title(f"Visualization - {current_filename}", fontsize=16)
            main_ax.axis('off')
            
            original_ax = save_fig.add_axes(original_pos)
            original_ax.set_title('Original Image', fontsize=14)
            original_ax.axis('off')
            
            mask_ax = save_fig.add_axes(mask_pos)
            mask_ax.set_title('Segmentation Mask', fontsize=14)
            mask_ax.axis('off')
            
            # Get current images and information
            original, mask, overlay, filename, object_data = self.load_and_process_image(self.txt_files[self.current_index])
            
            # Update title with actual filename
            main_ax.set_title(f"Visualization - {filename}", fontsize=16)
            
            # Plot main visualization based on current display mode
            if self.display_mode == "overlay":
                main_ax.imshow(overlay)
            elif self.display_mode == "side_by_side":
                composite = np.hstack((original, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255))
                main_ax.imshow(composite)
            elif self.display_mode == "masked":
                masked = original.copy()
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                masked = masked * mask_rgb
                main_ax.imshow(masked)
            
            # Show detailed panels
            original_ax.imshow(original)
            mask_ax.imshow(mask, cmap='gray')
            
            # Add a footer with timestamp
            plt.figtext(0.5, 0.01, f"Created: {timestamp}", ha='center', fontsize=10)
            
            # Save and close the figure
            save_fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(save_fig)
            
            print(f"Saved visualization as: {save_path}")
            
            # Simply update the display which will refresh the text labels
            self.set_section_titles()
            
            # Show a temporary message on the main display indicating successful save
            self.main_display_ax.text(0.5, 0.05, f"Saved as: {save_filename}", 
                          ha='center', va='bottom', color='yellow', fontsize=14,
                          transform=self.main_display_ax.transAxes,
                          bbox=dict(facecolor='black', alpha=0.7))
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
            # Show error message on the main display
            self.main_display_ax.text(0.5, 0.95, f"Error saving: {str(e)}", 
                          ha='center', va='top', color='red', fontsize=12,
                          transform=self.main_display_ax.transAxes,
                          bbox=dict(facecolor='black', alpha=0.7))
            self.fig.canvas.draw_idle()

    def mark_as_faulty(self, event=None):
        """Move current image and its label to faulty folder"""
        if not self.txt_files:
            return
            
        try:
            current_txt = self.txt_files[self.current_index]
            current_img = current_txt.replace('labels', 'images').replace('.txt', '.png')
            
            txt_filename = os.path.basename(current_txt)
            img_filename = os.path.basename(current_img)
            
            # Move files to faulty folder
            shutil.move(current_txt, os.path.join(self.faulty_labels, txt_filename))
            shutil.move(current_img, os.path.join(self.faulty_images, img_filename))
            
            # Remove from list and update display
            self.txt_files.pop(self.current_index)
            
            # Update statistics
            self.dataset_stats["faulty_count"] += 1
            self.dataset_stats["total_images"] = len(self.txt_files)
            
            # Update UI with success message and recalculate stats
            if len(self.txt_files) == 0:
                # Clear all texts and set a "no more images" message
                for txt in self.fig.texts[:]:
                    txt.remove()
                
                # Add a message in the center of the main display
                self.main_display_ax.text(0.5, 0.5, "No more images to display.", 
                              ha='center', va='center', color='white', fontsize=16,
                              transform=self.main_display_ax.transAxes)
                print("No more images to display.")
            else:
                # Adjust index if needed
                if self.current_index >= len(self.txt_files):
                    self.current_index = len(self.txt_files) - 1
                
                # Recalculate statistics
                self.calculate_dataset_stats()
                
                # Update display
                self.update_display()
                
                # Show success message
                print(f"Moved {img_filename} and {txt_filename} to faulty folder")
                
        except Exception as e:
            print(f"Error moving files to faulty folder: {str(e)}")
            # Show error message on the main display
            self.main_display_ax.text(0.5, 0.95, f"Error: {str(e)}", 
                          ha='center', va='top', color='red', fontsize=12,
                          transform=self.main_display_ax.transAxes,
                          bbox=dict(facecolor='black', alpha=0.7))
            self.fig.canvas.draw_idle()
    
    def next_image(self, event=None):
        """Show next image"""
        if self.current_index < len(self.txt_files) - 1:
            self.current_index += 1
            self.update_display()
    
    def previous_image(self, event=None):
        """Show previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard navigation"""
        if event.key == 'right':
            self.next_image()
        elif event.key == 'left':
            self.previous_image()
        elif event.key == 's':
            self.save_figure()
        elif event.key == 'f':
            self.mark_as_faulty()
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'o':
            # Toggle overlay mode
            self.display_mode = "overlay"
            self.update_display()
        elif event.key == 'm':
            # Toggle masked mode
            self.display_mode = "masked"
            self.update_display()
        elif event.key == 'b':
            # Toggle side-by-side mode
            self.display_mode = "side_by_side"
            self.update_display()
        elif event.key == 'c':
            # Toggle contours
            self.show_contours = not self.show_contours
            self.update_display()
        elif event.key == 'i':
            # Toggle class IDs
            self.show_class_ids = not self.show_class_ids
            self.update_display()
        elif event.key == 'h':
            # Show help text
            self.show_help_dialog()
    
    def show_help_dialog(self):
        """Display help information in a dialog"""
        help_text = """
        Keyboard Shortcuts:
        ------------------
        Right Arrow: Next image
        Left Arrow:  Previous image
        s:          Save visualization
        f:          Mark image as faulty
        o:          Overlay view mode
        m:          Masked view mode
        b:          Side-by-side view mode
        c:          Toggle contours
        i:          Toggle class IDs
        h:          Show this help
        q:          Quit
        
        Mouse Controls:
        --------------
        Use the UI controls at the bottom to:
        - Navigate between images
        - Adjust mask opacity
        - Change display mode
        - Toggle visual options
        """
        
        # Create a simple dialog figure
        help_fig = plt.figure(figsize=(8, 6))
        help_fig.canvas.manager.set_window_title('Help - SAM Annotation Visualizer')
        
        ax = help_fig.add_subplot(111)
        ax.text(0.5, 0.5, help_text, ha='center', va='center', 
                fontsize=12, multialignment='left',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.1))
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def export_dataset_stats(self, output_path=None):
        """Export dataset statistics to a JSON file"""
        if output_path is None:
            # Create default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.base_folder, f'dataset_stats_{timestamp}.json')
        
        try:
            # Prepare statistics for JSON serialization
            export_stats = self.dataset_stats.copy()
            
            # Convert Counter to regular dict for JSON serialization
            export_stats['class_distribution'] = dict(export_stats['class_distribution'])
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_stats, f, indent=4)
                
            print(f"Exported dataset statistics to {output_path}")
            return True
        except Exception as e:
            print(f"Error exporting statistics: {str(e)}")
            return False

def find_classes_csv(base_folder):
    """
    Attempt to find a classes CSV file in the category folder or parent directories.
    Returns the path if found, None otherwise.
    """
    # Check for a .sam_config.json file in the current directory
    if os.path.exists(".sam_config.json"):
        try:
            with open(".sam_config.json", 'r') as f:
                config = json.load(f)
                if "last_classes_csv" in config and os.path.exists(config["last_classes_csv"]):
                    return config["last_classes_csv"]
        except Exception:
            pass  # Silently continue if we can't read the config file
    
    # Check for any CSV files in the base folder
    for item in os.listdir(base_folder):
        if item.endswith('.csv'):
            csv_path = os.path.join(base_folder, item)
            try:
                # Try to read the CSV to verify it's a valid classes file
                df = pd.read_csv(csv_path)
                if 'class_name' in df.columns:
                    print(f"Found classes CSV file: {csv_path}")
                    return csv_path
            except Exception:
                pass  # Not a valid CSV file, continue searching
    
    # Check parent directory for common class CSV filenames
    parent_dir = os.path.dirname(base_folder)
    common_names = ['class_map.csv', 'classes.csv', 'class_map_sam.csv', 'class_names.csv']
    for name in common_names:
        path = os.path.join(parent_dir, name)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'class_name' in df.columns:
                    print(f"Found classes CSV file: {path}")
                    return path
            except Exception:
                pass
    
    return None

def view_masks(base_folder, export_stats=False, classes_csv=None):
    """
    Launch the interactive multi-object mask viewer
    
    Args:
        base_folder: Path to the folder containing 'images' and 'labels' subfolders
        export_stats: Whether to export dataset statistics to a JSON file
        classes_csv: Optional path to CSV file with class names
    """
    try:
        print(f"Loading annotations from: {base_folder}")
        
        # If no classes CSV is provided, try to find one
        if not classes_csv:
            found_csv = find_classes_csv(base_folder)
            if found_csv:
                print(f"Found classes CSV file: {found_csv}")
                classes_csv = found_csv
        
        viewer = MultiMaskViewer(base_folder, classes_csv=classes_csv)
        
        # Export statistics if requested
        if export_stats:
            stats_path = os.path.join(base_folder, f'dataset_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            if viewer.export_dataset_stats(stats_path):
                print(f"Exported dataset statistics to {stats_path}")
        
        print("Controls:")
        print("  Right/Left Arrows: Navigate between images")
        print("  's' key: Save visualization")
        print("  'f' key: Mark as faulty")
        print("  'h' key: Display help")
        print("  'q' key: Quit")
        plt.show()
    except Exception as e:
        print(f"Error initializing viewer: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM Annotation Visualizer")
    parser.add_argument('--folder', type=str, default='test_data/s3',
                       help="Path to the folder containing 'images' and 'labels' subfolders")
    parser.add_argument('--export_stats', action='store_true',
                       help="Export dataset statistics to JSON file")
    parser.add_argument('--classes_csv', type=str, default=None,
                       help="Path to CSV file containing class names (must have a 'class_name' column)")
    
    args = parser.parse_args()
    
    # Use the provided folder path and normalize for cross-platform compatibility
    base_folder = os.path.normpath(args.folder)
    
    # Verify the folder structure
    images_dir = os.path.join(base_folder, 'images')
    labels_dir = os.path.join(base_folder, 'labels')
    
    if not os.path.exists(base_folder):
        print(f"Error: Base folder '{base_folder}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print("Available directories:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
        sys.exit(1)
        
    if not os.path.exists(images_dir):
        print(f"Error: Images folder '{images_dir}' not found.")
        print(f"Creating images folder...")
        os.makedirs(images_dir, exist_ok=True)
        
    if not os.path.exists(labels_dir):
        print(f"Error: Labels folder '{labels_dir}' not found.")
        print(f"Creating labels folder...")
        os.makedirs(labels_dir, exist_ok=True)
    
    # Check if there are any annotation files in the labels folder
    annotation_files = glob(os.path.join(labels_dir, '*.txt'))
    if not annotation_files:
        print(f"Warning: No annotation files (*.txt) found in '{labels_dir}'.")
        print("The viewer will start but may not display any images.")
    
    # Create the viewer
    try:
        view_masks(base_folder, export_stats=args.export_stats, classes_csv=args.classes_csv)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        
        
# export MPLBACKEND=TkAgg

