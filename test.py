
import torch
import sys 

def check_torch_cuda():
    # Print PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Print CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Print current device information
        current_device = torch.cuda.current_device()
        print(f"Current CUDA Device: {current_device}")
        print(f"Device Name: {torch.cuda.get_device_name(current_device)}")
        
        # Print device capability
        capability = torch.cuda.get_device_capability(current_device)
        print(f"Device Capability: {capability[0]}.{capability[1]}")
        
        # Print maximum memory allocated
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    check_torch_cuda()
sys.exit()

# Removing duplicate functions

def _on_annotation_delete(self, idx: int) -> None:
    """Handle annotation deletion."""
    if 0 <= idx < len(self.annotations):
        del self.annotations[idx]
        self.window_manager.update_review_panel(self.annotations)


""" 
   COMING UP TASKS
        1. Fix the save annotation. We should save the full mask
        2. Fix the visualization of masks and the overlay, add features
        3. 
"""

 

# Work hard to improve this class 
""" 
    Main Class SamAnnotator Predictor: 
        List of Methods:
            1. __init__
            2. _initialize_model 
            3. _load_classes 
            4. _setup_paths 
            5. _setup_callbacks 
            6. _on_annotation_select 
            7. _on_annotation_class_change
            8. _handle_mask_prediction
            9. _handle_class_selection 
            10. _load_image
            11. _add_annotation
            12. _save_annotation
            13. _prev_image
            14. _next_image
            15. _remove_last_annotation
            16. _get_label_path 
            17. _save_annotations_to_file
            18. on_annotation_delete
            19. _handle_undo
            20. _handle_redo
            21. _handle_export
            22. _get_last_annotated_index
            23. _create_predictor
            24. _load_annotation
            25. run

    ui - class EventHandler
        List of Methods
            1. __init__
            2. register_callbacks
            3. handle_mouse_event
            4. handle_class_window_event
            5. handle_keyboard_event
            6. reset_state

    ui  - class WindowManager
        List of Methods
            1. __init__
            2. setup_windows
            3. _handle_view_state_change
            4. update_main_window
            5. update_review_panel
            6. handle_review_keyboard
            7.handle_keyboard_event
            8. get_selected_annotation_idx
            9. update_class_window
            10. set_mask
            11. toggle_view_option
            11. set_image_scale
            12. destroy_windows

    ui/widget - Class ClassSelector
        List of Methods
            1. __init__
            2. set_classes
            3. set_current_class
            4. _update_window_size
            5. _get_class_at_position
            6. handle_mouse
            7. render
            8. destroy

    ui/widget - Class StatusOverlay
        List of Methods
            1. __init__
            2. _add_text_with_background
            3. render

    ui/widget - Class AnnotationReview
        List of Methods
            1. __init__
            2. toggle_visibility
            3. set_mouse_callback
            4. set_annotations
            5. register_callbacks
            6. _update_panel_size
            7. draw_header
            8. draw_toolbar 
            9. _draw_annotation_entry
            10. handle_keyboard
            11. handle_mouse   
            12. render
            13. destroy

    ui/widget - Class ViewControls
        List of Methods
            1. __init__
            2. _notify_state_change
            3. toggle_state
            4. register_callback
            5. _create_control_panel
            6. _draw_slider
            7. _draw_toggle
            8. _get_slider_value
            9. update_state
            10. toggle_visibility
            11. handle_keyboard
            12. handle_mouse
            13. render
            14. get_state
            15. destroy

    utils/ class VisualizationManager
        List of Methods
            1. __init__
            2. _generate_colors
            3. _get_text_color
            4. _draw_mask 
            5. _draw_box
            6. _draw_label
            7. _draw_points
            8. set_color_scheme
            9. set_mask_opacity 
            10. add_grid_overlay 
            11. create_minimap
            12. create_side_by_side_view
            13. highlight_overlapping_regions
            14. create_measurement_overlay
            15. create_annotation_preview
            16. _add_text_with_background
            17. create_composite_view
            18. add_status_overlay
    utils/ class ImageProcessor
        List of Methods
            1. __init__
            2. process_image
            3. _create_metadata
            4. scale_coordinates_to_original
            5. scale_coordinates_to_display
            6. scale_box_to_original
            7. scale_box_to_display
            8. scale_contour_to_original
            9. scale_contour_to_display
            
    config/ shortcuts.py 
    config/ settings.py 
        

"""