from typing import List, Tuple, Optional, Callable
import cv2
import numpy as np
from ...config.settings import BUTTON_HEIGHT, MAX_VISIBLE_CLASSES

class ClassSelector:
    """Widget for class selection interface with scrollable support for 1000+ classes."""

    def __init__(self, window_name: str, width: int = 200):
        """
        Initialize the class selector widget.

        Args:
            window_name (str): Name of the window to create
            width (int): Width of the class selector window
        """
        self.window_name = window_name
        self.width = width
        self.button_height = BUTTON_HEIGHT
        self.max_visible_classes = MAX_VISIBLE_CLASSES
        self.classes: List[str] = []
        self.current_class_id: int = 0
        self.hover_idx: Optional[int] = None
        self.scroll_offset: int = 0  # Tracks which class is at the top of the visible window

        # Create window
        cv2.namedWindow(self.window_name)

    def set_classes(self, classes: List[str]) -> None:
        """Set available classes."""
        self.classes = classes
        self.scroll_offset = 0  # Reset scroll when classes change
        self._update_window_size()

    def set_current_class(self, class_id: int) -> None:
        """Set current selected class and auto-scroll to it if needed."""
        if 0 <= class_id < len(self.classes):
            self.current_class_id = class_id
            # Auto-scroll to make selected class visible
            if class_id < self.scroll_offset:
                self.scroll_offset = class_id
            elif class_id >= self.scroll_offset + self.max_visible_classes:
                self.scroll_offset = class_id - self.max_visible_classes + 1

    def _update_window_size(self) -> None:
        """Update window size based on visible classes (not total classes)."""
        visible_count = min(len(self.classes), self.max_visible_classes)
        height = visible_count * self.button_height
        self.window_image = np.zeros((height, self.width, 3), dtype=np.uint8)

    def _get_class_at_position(self, y: int) -> Optional[int]:
        """Get class index at given y coordinate (accounting for scroll offset)."""
        visible_idx = y // self.button_height
        actual_idx = visible_idx + self.scroll_offset
        if 0 <= actual_idx < len(self.classes):
            return actual_idx
        return None

    def _scroll(self, delta: int) -> None:
        """
        Scroll the class list.

        Args:
            delta: Positive to scroll down, negative to scroll up
        """
        old_offset = self.scroll_offset
        self.scroll_offset = max(0, min(
            len(self.classes) - self.max_visible_classes,
            self.scroll_offset + delta
        ))

        # Only render if scroll actually changed
        if old_offset != self.scroll_offset:
            self.render()

    def handle_keyboard(self, key: int) -> Optional[int]:
        """
        Handle keyboard events for class navigation and selection.

        Args:
            key: Key code from cv2.waitKey()

        Returns:
            New selected class_id if a selection was made, None otherwise
        """
        if not self.classes:
            return None

        # Arrow key codes can vary by system and OpenCV version
        # Common codes: Up=82/2490368/65362, Down=84/2621440/65364
        #               Left=81/2424832/65361, Right=83/2555904/65363
        #               Page Up=85/2162688/65365, Page Down=86/2228224/65366
        #               Home=71/2359296, End=79/2293760

        # Mask high bits for special keys on some systems
        key_masked = key & 0xFF

        new_class_id = None

        # Arrow Up - select previous class
        if key_masked == 82 or key == 2490368 or key == 65362:
            new_class_id = max(0, self.current_class_id - 1)

        # Arrow Down - select next class
        elif key_masked == 84 or key == 2621440 or key == 65364:
            new_class_id = min(len(self.classes) - 1, self.current_class_id + 1)

        # Page Up - select class 5 positions up
        elif key_masked == 85 or key == 2162688 or key == 65365:
            new_class_id = max(0, self.current_class_id - 5)

        # Page Down - select class 5 positions down
        elif key_masked == 86 or key == 2228224 or key == 65366:
            new_class_id = min(len(self.classes) - 1, self.current_class_id + 5)

        # Home - select first class
        elif key_masked == 71 or key == 2359296:
            new_class_id = 0

        # End - select last class
        elif key_masked == 79 or key == 2293760:
            new_class_id = len(self.classes) - 1

        # If a new class was selected, update and auto-scroll
        if new_class_id is not None and new_class_id != self.current_class_id:
            self.set_current_class(new_class_id)
            self.render()
            return new_class_id

        return None

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> Optional[int]:
        """
        Handle mouse events including scroll wheel.

        Returns:
            Selected class index if a selection was made, None otherwise
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self._get_class_at_position(y)
            self.render()

        elif event == cv2.EVENT_LBUTTONDOWN:
            selected_idx = self._get_class_at_position(y)
            if selected_idx is not None:
                self.current_class_id = selected_idx
                self.render()
                return selected_idx

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll with mouse wheel
            # flags > 0 means scroll up, < 0 means scroll down
            scroll_delta = -1 if flags > 0 else 1
            self._scroll(scroll_delta * 3)  # Scroll 3 items at a time

        return None

    def render(self) -> None:
        """Render the class selector window showing only visible classes."""
        self.window_image.fill(0)  # Clear the image

        # Calculate visible range
        visible_start = self.scroll_offset
        visible_end = min(len(self.classes), self.scroll_offset + self.max_visible_classes)

        for i in range(visible_start, visible_end):
            # Calculate position in visible window
            visible_idx = i - visible_start
            y = visible_idx * self.button_height

            # Determine button color
            if i == self.current_class_id:
                color = (0, 255, 0)  # Selected - Green
            elif i == self.hover_idx:
                color = (100, 100, 100)  # Hover - Gray
            else:
                color = (50, 50, 50)  # Normal - Dark Gray

            # Draw button background
            cv2.rectangle(self.window_image,
                         (0, y),
                         (self.width, y + self.button_height),
                         color, -1)

            # Draw class name
            class_text = f"{i}: {self.classes[i]}"
            # Truncate if too long
            if len(class_text) > 25:
                class_text = class_text[:22] + "..."

            cv2.putText(self.window_image,
                       class_text,
                       (5, y + self.button_height - 8),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,  # Slightly smaller font for better fit
                       (255, 255, 255),  # White text
                       1)

            # Draw separator line
            cv2.line(self.window_image,
                    (0, y + self.button_height - 1),
                    (self.width, y + self.button_height - 1),
                    (0, 0, 0), 1)

        # Draw scroll indicator if there are more classes above or below
        if self.scroll_offset > 0:
            # Draw "more above" indicator (up arrow)
            cv2.putText(self.window_image,
                       "^ More above",
                       (self.width - 90, 15),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.4,
                       (255, 255, 0),  # Yellow
                       1)

        if visible_end < len(self.classes):
            # Draw "more below" indicator (down arrow)
            bottom_y = (visible_end - visible_start) * self.button_height - 5
            cv2.putText(self.window_image,
                       "v More below",
                       (self.width - 95, bottom_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.4,
                       (255, 255, 0),  # Yellow
                       1)

        # Draw class count info at top-left
        info_text = f"{len(self.classes)} classes"
        cv2.putText(self.window_image,
                   info_text,
                   (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4,
                   (200, 200, 200),  # Light gray
                   1)

        cv2.imshow(self.window_name, self.window_image)

    def destroy(self) -> None:
        """Destroy the class selector window."""
        try:
            # Check if window exists by trying to get its property
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except:
            pass  # Window doesn't exist or already destroyed
