"""Keyboard shortcuts configuration"""

# Basic navigation
SHORTCUTS = {
    'quit': 'q',
    'next_image': 'n',
    'prev_image': 'p',
    'save': 's',
    'clear_selection': 'x', 
    'add_annotation': 'a',
    'undo': 'z',  # Add undo shortcut
    'redo': 'y',  # Add redo shortcut
    'clear_all': 'c',  
    
    # View controls
    'toggle_masks': 'm',
    'toggle_boxes': 'b',
    'toggle_labels': 'l',
    'toggle_points': 't',
    'toggle_view_controls': 'v',
     
    # Toggle annotation review
    'toggle_review':'r', 
    
    # Export shortcuts (new)
    'export_coco': 'e',  # Press 'e' then 'c' for COCO export
    'export_yolo': 'e',  # Press 'e' then 'y' for YOLO export
    'export_pascal': 'e', # Press e then 'p' for pascal export     
    
    # Annotation modes
    'toggle_mode': 'w',  # Toggle between box and point annotation modes
    
    # Zoom controls
    'zoom_in': '=',    # Plus key
    'zoom_out': '-',   # Minus key
    'zoom_reset': '0', # Reset zoom to 100%
    
    # Opacity controls
    'opacity_up': ']',   # Increase opacity
    'opacity_down': '[', # Decrease opacity
}

# Function key shortcuts (if needed)
FUNCTION_SHORTCUTS = { 
    'F1': 'help',
    'F2': 'save_view_settings',
    'F3': 'load_view_settings'
}