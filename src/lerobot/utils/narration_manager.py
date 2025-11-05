"""
Narration Manager for dataset recording.

This module provides the NarrationManager class that manages narration text during dataset recording.
"""


class NarrationManager:
    """
    Manages narration text for dataset recording.

    This class handles a list of narration strings, allowing them to be popped in order
    and maintaining a history of previous narrations. It's designed to be used during
    robot dataset recording to track task descriptions over time.

    Attributes:
        narrations: List of remaining narration strings to be used.
        current_narration: The currently active narration string.
        previous_narrations: List of narration strings that have been used.
        max_history: Maximum number of previous narrations to keep in history.
    """

    def __init__(self, narrations: list[str] | None = None, max_history: int = 5):
        """
        Initialize the NarrationManager.

        Args:
            narrations: List of narration strings. If None, narration functionality is disabled.
            max_history: Maximum number of previous narrations to keep in history (default: 5).
        """
        self.narrations = list(narrations) if narrations else []
        self.current_narration = ""
        self.previous_narrations: list[str] = []
        self.max_history = max_history
        self._enabled = narrations is not None and len(narrations) > 0

    def is_enabled(self) -> bool:
        """
        Check if narration functionality is enabled.

        Returns:
            True if narrations were provided and the feature is active, False otherwise.
        """
        return self._enabled

    def has_narrations(self) -> bool:
        """
        Check if there are remaining narrations to be used.

        Returns:
            True if there are narrations left in the queue, False otherwise.
        """
        return len(self.narrations) > 0

    def pop_narration(self) -> str:
        """
        Pop the next narration from the list and make it the current narration.

        If there's already a current narration, it's moved to the previous narrations history
        before popping the next one. The history is limited to max_history items.

        Returns:
            The new current narration string, or empty string if no narrations remain.
        """
        if not self.has_narrations():
            return ""

        # Move current narration to history
        if self.current_narration:
            self.previous_narrations.append(self.current_narration)
            # Limit history size
            if len(self.previous_narrations) > self.max_history:
                self.previous_narrations.pop(0)

        # Pop next narration
        self.current_narration = self.narrations.pop(0)
        return self.current_narration

    def get_current_narration(self) -> str:
        """
        Get the current narration string.

        Returns:
            The current narration string.
        """
        return self.current_narration

    def get_previous_narrations(self) -> list[str]:
        """
        Get the list of previous narrations.

        Returns:
            List of previous narration strings.
        """
        return self.previous_narrations.copy()

    def get_previous_narrations_as_string(self) -> str:
        """
        Get previous narrations as a newline-separated string.

        This format is used for storing in the dataset.

        Returns:
            Newline-separated string of previous narrations.
        """
        return "\n".join(self.previous_narrations)

    def get_next_narration_preview(self) -> str:
        """
        Get a preview of the next narration without popping it.

        Returns:
            The next narration string if available, empty string otherwise.
        """
        return self.narrations[0] if self.has_narrations() else ""

    def get_remaining_count(self) -> int:
        """
        Get the number of remaining narrations.

        Returns:
            Number of narrations still in the queue.
        """
        return len(self.narrations)

    def reset(self) -> None:
        """
        Reset the narration manager state.

        Clears current narration and previous narrations history.
        Does not reset the narrations queue.
        """
        self.current_narration = ""
        self.previous_narrations = []

    def should_end_episode(self) -> bool:
        """
        Check if the episode should end based on narration state.

        When narration mode is enabled, the episode should end when all narrations
        have been used.

        Returns:
            True if narration mode is enabled and no narrations remain, False otherwise.
        """
        return self.is_enabled() and not self.has_narrations()
