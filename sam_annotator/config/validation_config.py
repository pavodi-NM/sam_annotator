# src/config/validation_config.py
DEFAULT_VALIDATION_RULES = {
    'min_size': 100,
    'max_overlap': 0.3,
    'required_fields': ['class_id', 'contour_points', 'box'],
    'max_annotations_per_class': 50,
    'min_annotations_per_class': 1
}