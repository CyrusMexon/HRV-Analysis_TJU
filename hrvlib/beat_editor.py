"""
beat_editor.py - Manual Beat Correction System
Implements SRS requirements FR-10 to FR-13 for manual editing tools
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EditAction(Enum):
    """Types of beat editing actions (SRS FR-10)"""

    DELETE = "delete"
    MOVE = "move"
    INTERPOLATE = "interpolate"
    UNDO = "undo"


class InterpolationMethod(Enum):
    """Interpolation methods (SRS FR-11)"""

    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"


@dataclass
class BeatEdit:
    """Record of a single beat edit for audit trail (SRS FR-13)"""

    timestamp: datetime
    action: EditAction
    beat_index: int
    original_value: Optional[float]
    new_value: Optional[float]
    user_id: str = "user"
    interpolation_method: Optional[InterpolationMethod] = None
    moved_from_index: Optional[int] = None  # For move operations


class BeatEditor:
    """
    Manual beat editing system implementing SRS FR-10 to FR-13

    Features:
    - Delete beats (FR-10)
    - Move beats to new positions (FR-10)
    - Interpolate missing/ectopic beats (FR-10)
    - Linear and cubic spline interpolation (FR-11)
    - Complete audit trail logging (FR-13)
    - Undo/redo functionality
    """

    def __init__(self, rr_intervals: List[float], user_id: str = "user"):
        """
        Initialize beat editor with RR interval data

        Args:
            rr_intervals: List of RR intervals in milliseconds
            user_id: User identifier for audit trail
        """
        self.original_rr = np.array(rr_intervals, dtype=float)
        self.current_rr = self.original_rr.copy()
        self.user_id = user_id

        # Audit trail (SRS FR-13)
        self.edit_history: List[BeatEdit] = []
        self.undo_stack: List[BeatEdit] = []

        # Edit statistics for quality reporting
        self.total_edits = 0
        self.deleted_beats = 0
        self.moved_beats = 0
        self.interpolated_beats = 0

    def delete_beat(self, beat_index: int) -> bool:
        """
        Delete a beat at specified index (SRS FR-10)

        Args:
            beat_index: Index of beat to delete (0-based)

        Returns:
            bool: True if deletion successful, False otherwise
        """
        if not self._validate_index(beat_index):
            return False

        # Record original value for undo
        original_value = self.current_rr[beat_index]

        # Perform deletion
        self.current_rr = np.delete(self.current_rr, beat_index)

        # Log edit in audit trail (SRS FR-13)
        edit = BeatEdit(
            timestamp=datetime.now(),
            action=EditAction.DELETE,
            beat_index=beat_index,
            original_value=original_value,
            new_value=None,
            user_id=self.user_id,
        )

        self._log_edit(edit)
        self.deleted_beats += 1

        return True

    def move_beat(self, from_index: int, to_index: int) -> bool:
        """
        Move a beat from one position to another (SRS FR-10)

        Args:
            from_index: Current index of beat to move
            to_index: Target index for beat

        Returns:
            bool: True if move successful, False otherwise
        """
        if (
            not self._validate_index(from_index)
            or to_index < 0
            or to_index >= len(self.current_rr)
        ):
            return False

        if from_index == to_index:
            return True  # No change needed

        # Get the beat value to move
        beat_value = self.current_rr[from_index]

        # Remove from original position
        self.current_rr = np.delete(self.current_rr, from_index)

        # Adjust target index if necessary
        if to_index > from_index:
            to_index -= 1

        # Insert at new position
        self.current_rr = np.insert(self.current_rr, to_index, beat_value)

        # Log edit in audit trail (SRS FR-13)
        edit = BeatEdit(
            timestamp=datetime.now(),
            action=EditAction.MOVE,
            beat_index=to_index,
            original_value=beat_value,
            new_value=beat_value,
            user_id=self.user_id,
            moved_from_index=from_index,
        )

        self._log_edit(edit)
        self.moved_beats += 1

        return True

    def interpolate_beat(
        self,
        beat_index: int,
        method: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE,
    ) -> bool:
        """
        Interpolate a beat value using specified method (SRS FR-10, FR-11)

        Args:
            beat_index: Index of beat to interpolate
            method: Interpolation method (linear or cubic spline)

        Returns:
            bool: True if interpolation successful, False otherwise
        """
        if not self._validate_index(beat_index):
            return False

        original_value = self.current_rr[beat_index]

        if method == InterpolationMethod.LINEAR:
            new_value = self._linear_interpolation(beat_index)
        elif method == InterpolationMethod.CUBIC_SPLINE:
            new_value = self._cubic_spline_interpolation(beat_index)
        else:
            return False

        if new_value is None:
            return False

        # Apply interpolated value
        self.current_rr[beat_index] = new_value

        # Log edit in audit trail (SRS FR-13)
        edit = BeatEdit(
            timestamp=datetime.now(),
            action=EditAction.INTERPOLATE,
            beat_index=beat_index,
            original_value=original_value,
            new_value=new_value,
            user_id=self.user_id,
            interpolation_method=method,
        )

        self._log_edit(edit)
        self.interpolated_beats += 1

        return True

    def insert_beat(
        self,
        insert_index: int,
        method: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE,
    ) -> bool:
        """
        Insert a new beat using interpolation (for missed beats)

        Args:
            insert_index: Index where to insert new beat
            method: Interpolation method

        Returns:
            bool: True if insertion successful, False otherwise
        """
        if insert_index < 0 or insert_index > len(self.current_rr):
            return False

        # Calculate interpolated value based on neighbors
        if method == InterpolationMethod.LINEAR:
            new_value = self._linear_interpolation_for_insertion(insert_index)
        else:
            new_value = self._cubic_spline_interpolation_for_insertion(insert_index)

        if new_value is None:
            return False

        # Insert the new beat
        self.current_rr = np.insert(self.current_rr, insert_index, new_value)

        # Log edit in audit trail
        edit = BeatEdit(
            timestamp=datetime.now(),
            action=EditAction.INTERPOLATE,  # Insertion is a type of interpolation
            beat_index=insert_index,
            original_value=None,  # New beat, no original value
            new_value=new_value,
            user_id=self.user_id,
            interpolation_method=method,
        )

        self._log_edit(edit)
        self.interpolated_beats += 1

        return True

    def undo_last_edit(self) -> bool:
        """
        Undo the last edit operation

        Returns:
            bool: True if undo successful, False if no edits to undo
        """
        if not self.edit_history:
            return False

        last_edit = self.edit_history.pop()

        # Reverse the operation
        if last_edit.action == EditAction.DELETE:
            # Re-insert the deleted beat
            self.current_rr = np.insert(
                self.current_rr, last_edit.beat_index, last_edit.original_value
            )
            self.deleted_beats -= 1

        elif last_edit.action == EditAction.MOVE:
            # Move beat back to original position
            beat_value = self.current_rr[last_edit.beat_index]
            self.current_rr = np.delete(self.current_rr, last_edit.beat_index)
            self.current_rr = np.insert(
                self.current_rr, last_edit.moved_from_index, beat_value
            )
            self.moved_beats -= 1

        elif last_edit.action == EditAction.INTERPOLATE:
            if last_edit.original_value is not None:
                # Restore original value
                self.current_rr[last_edit.beat_index] = last_edit.original_value
                self.interpolated_beats -= 1
            else:
                # This was an insertion, remove the beat
                self.current_rr = np.delete(self.current_rr, last_edit.beat_index)
                self.interpolated_beats -= 1

        # Add to undo stack for potential redo
        self.undo_stack.append(last_edit)
        self.total_edits -= 1

        return True

    def reset_all_edits(self) -> bool:
        """
        Reset all edits and restore original RR intervals

        Returns:
            bool: True if reset successful
        """
        self.current_rr = self.original_rr.copy()
        self.edit_history.clear()
        self.undo_stack.clear()

        # Reset statistics
        self.total_edits = 0
        self.deleted_beats = 0
        self.moved_beats = 0
        self.interpolated_beats = 0

        return True

    def get_corrected_beats_percentage(self) -> float:
        """
        Calculate percentage of corrected beats (SRS FR-14)

        Returns:
            float: Percentage of beats that were corrected
        """
        if len(self.original_rr) == 0:
            return 0.0

        return (self.total_edits / len(self.original_rr)) * 100.0

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """
        Get complete audit trail for export (SRS FR-13)

        Returns:
            List[Dict]: List of edit records with timestamps, actions, and user info
        """
        trail = []
        for edit in self.edit_history:
            record = {
                "timestamp": edit.timestamp.isoformat(),
                "action": edit.action.value,
                "beat_index": edit.beat_index,
                "original_value": edit.original_value,
                "new_value": edit.new_value,
                "user": edit.user_id,
            }

            if edit.interpolation_method:
                record["interpolation_method"] = edit.interpolation_method.value

            if edit.moved_from_index is not None:
                record["moved_from_index"] = edit.moved_from_index

            trail.append(record)

        return trail

    def get_edit_statistics(self) -> Dict[str, Any]:
        """
        Get editing statistics for quality assessment

        Returns:
            Dict: Statistics about edits performed
        """
        return {
            "total_edits": self.total_edits,
            "deleted_beats": self.deleted_beats,
            "moved_beats": self.moved_beats,
            "interpolated_beats": self.interpolated_beats,
            "corrected_beats_percentage": self.get_corrected_beats_percentage(),
            "original_beat_count": len(self.original_rr),
            "current_beat_count": len(self.current_rr),
        }

    def _validate_index(self, index: int) -> bool:
        """Validate that index is within current RR interval bounds"""
        return 0 <= index < len(self.current_rr)

    def _log_edit(self, edit: BeatEdit) -> None:
        """Log edit in audit trail and update statistics"""
        self.edit_history.append(edit)
        self.total_edits += 1

        # Clear undo stack when new edit is made
        self.undo_stack.clear()

    def _linear_interpolation(self, index: int) -> Optional[float]:
        """
        Perform linear interpolation for beat at index (SRS FR-11)
        """
        if len(self.current_rr) < 3:
            return None

        # Use neighbors for interpolation
        if index == 0:
            # Use next two values
            return (self.current_rr[1] + self.current_rr[2]) / 2.0
        elif index == len(self.current_rr) - 1:
            # Use previous two values
            return (self.current_rr[index - 1] + self.current_rr[index - 2]) / 2.0
        else:
            # Use previous and next values
            return (self.current_rr[index - 1] + self.current_rr[index + 1]) / 2.0

    def _cubic_spline_interpolation(self, index: int) -> Optional[float]:
        """
        Perform cubic spline interpolation for beat at index (SRS FR-11)
        """
        if len(self.current_rr) < 4:
            # Fall back to linear interpolation
            return self._linear_interpolation(index)

        try:
            from scipy.interpolate import CubicSpline

            # Create array without the current point
            indices = np.arange(len(self.current_rr))
            mask = indices != index
            x_interp = indices[mask]
            y_interp = self.current_rr[mask]

            # Create cubic spline
            cs = CubicSpline(x_interp, y_interp)

            # Interpolate at the target index
            return float(cs(index))

        except ImportError:
            # Fall back to linear interpolation if scipy not available
            return self._linear_interpolation(index)
        except Exception:
            # Fall back to linear interpolation on any error
            return self._linear_interpolation(index)

    def _linear_interpolation_for_insertion(self, insert_index: int) -> Optional[float]:
        """Linear interpolation for inserting new beat"""
        if len(self.current_rr) < 2:
            return None

        if insert_index == 0:
            return self.current_rr[0]
        elif insert_index >= len(self.current_rr):
            return self.current_rr[-1]
        else:
            return (
                self.current_rr[insert_index - 1] + self.current_rr[insert_index]
            ) / 2.0

    def _cubic_spline_interpolation_for_insertion(
        self, insert_index: int
    ) -> Optional[float]:
        """Cubic spline interpolation for inserting new beat"""
        if len(self.current_rr) < 3:
            return self._linear_interpolation_for_insertion(insert_index)

        try:
            from scipy.interpolate import CubicSpline

            # Create spline from existing data
            x = np.arange(len(self.current_rr))
            cs = CubicSpline(x, self.current_rr)

            # Interpolate at insertion point (between existing indices)
            interp_x = insert_index - 0.5 if insert_index > 0 else 0.5
            return float(cs(interp_x))

        except ImportError:
            return self._linear_interpolation_for_insertion(insert_index)
        except Exception:
            return self._linear_interpolation_for_insertion(insert_index)

    def get_current_rr_intervals(self) -> np.ndarray:
        """Get current (edited) RR intervals"""
        return self.current_rr.copy()

    def get_original_rr_intervals(self) -> np.ndarray:
        """Get original (unedited) RR intervals"""
        return self.original_rr.copy()
