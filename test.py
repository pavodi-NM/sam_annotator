

# Removing duplicate functions

def _on_annotation_delete(self, idx: int) -> None:
    """Handle annotation deletion."""
    if 0 <= idx < len(self.annotations):
        del self.annotations[idx]
        self.window_manager.update_review_panel(self.annotations)


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

"""