# src/core/base_annotator.py
class BaseAnnotator:
    """
    Core annotator class that manages the annotation process.
    Delegates task-specific operations to the appropriate task handler.
    """
    
    def __init__(self, task_handler: BaseAnnotationTask, config: Dict[str, Any]):
        self.task_handler = task_handler
        self.config = config
        self.current_annotation = None
        self.history = []
        
    def handle_interaction(self, x: float, y: float, interaction_type: str) -> Dict[str, Any]:
        """Handle user interaction and delegate to task handler"""
        result = self.task_handler.process_interaction(x, y, interaction_type)
        if result.get('annotation_complete', False):
            self.current_annotation = result.get('annotation')
            self.history.append(self.current_annotation)
        return result