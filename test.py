

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
            19. run

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
            7. get_selected_annotation_idx
            8. update_class_window
            8. set_mask
            9. toggle_view_option
            10. set_image_scale
            11. destroy_windows

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
            4. 

    ui/widget - Class AnnotationReview
        List of Methods
            1. __init__
            2. set_annotations
            3. register_callbacks
            4. _update_panel_size
            5. _draw_annotation_entry
            6. handle_mouse
            7. handle_keyboard
            8. render
            9. destroy

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
            11. handle_mouse
            12. render
            13. get_state
            14. destroy

    utils/ class VisualizationManager
        List of Methods
            1. __init__
            2. _generate_colors
            3. _draw_mask
            4. draw_box 
            5. _draw_label
            6. _draw_points
            7. set_color_scheme
            8. set_mask_opacity 
            9. add_grid_overlay 
            10. create_minimap
            11. create_side_by_side_view
            12. highlight_overlapping_regions
            13. create_measurement_overlay
            14. create_annotation_preview
            15. create_composite_view
            16. add_status_overlay
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