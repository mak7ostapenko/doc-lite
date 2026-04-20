"""
Coordinate system data structures for OCR pipeline.

This module provides explicit coordinate tracking to prevent transformation errors.
Extracted from general_ocr_pipeline.py to avoid circular imports.
"""

from enum import Enum
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# COORDINATE SYSTEM DATA STRUCTURES
# ============================================================================

class CoordinateSystem(Enum):
    """
    Explicit coordinate system tracking to prevent transformation errors.

    PAGE_ABSOLUTE: Coordinates relative to preprocessed document image (after orientation/unwarp)
    REGION_RELATIVE: Coordinates relative to cropped region (table, seal, etc.)
    """
    PAGE_ABSOLUTE = "page"
    REGION_RELATIVE = "region"


@dataclass(frozen=True)
class BBox:
    """
    Immutable bounding box with explicit coordinate system tracking.

    Replaces ad-hoc [x1,y1,x2,y2] lists and {"coordinate": [...], "label": ...} dicts.

    Attributes:
        x_min, y_min, x_max, y_max: Box coordinates (axis-aligned rectangle)
        coord_system: Which coordinate frame this box is in
        label: Optional semantic label (e.g., "text", "table", "formula")
        confidence: Optional detection/recognition confidence score
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    coord_system: CoordinateSystem
    label: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        """Validate box coordinates."""
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError(f"Invalid bbox: ({self.x_min},{self.y_min}) to ({self.x_max},{self.y_max})")

    @cached_property
    def width(self) -> float:
        return self.x_max - self.x_min

    @cached_property
    def height(self) -> float:
        return self.y_max - self.y_min

    @cached_property
    def area(self) -> float:
        return self.width * self.height

    @cached_property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_list(self) -> List[float]:
        """Convert to legacy [x1,y1,x2,y2] format."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_dict(self) -> dict:
        """Convert to legacy DetResult format."""
        return {
            "coordinate": self.to_list(),
            "label": self.label,
            "score": self.confidence if self.confidence is not None else 1.0
        }

    def __repr__(self) -> str:
        label_str = f" label={self.label}" if self.label else ""
        return f"BBox([{self.x_min:.1f}, {self.y_min:.1f}, {self.x_max:.1f}, {self.y_max:.1f}] {self.coord_system.value}{label_str})"


@dataclass(frozen=True)
class BPolygon:
    """
    Polygon representation (typically 4-point for quadrilaterals).

    Used for text detection results that may be rotated rectangles.

    Attributes:
        points: Numpy array shape (N, 2) where N is typically 4
        coord_system: Which coordinate frame these points are in
        label: Optional semantic label
        confidence: Optional confidence score
    """
    points: np.ndarray  # Shape (N, 2)
    coord_system: CoordinateSystem
    label: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        """Validate polygon structure."""
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"Points must be (N, 2) array, got shape {self.points.shape}")
        if len(self.points) < 3:
            raise ValueError(f"Polygon must have at least 3 points, got {len(self.points)}")

    def to_bbox(self) -> BBox:
        """Convert polygon to axis-aligned bounding box."""
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        return BBox(
            x_min=float(np.min(x_coords)),
            y_min=float(np.min(y_coords)),
            x_max=float(np.max(x_coords)),
            y_max=float(np.max(y_coords)),
            coord_system=self.coord_system,
            label=self.label,
            confidence=self.confidence
        )

    @cached_property
    def area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        x = self.points[:, 0]
        y = self.points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@dataclass
class CoordinateTransform:
    """
    Explicit coordinate system transformation.

    Tracks frame-to-frame transformations (e.g., page → region, region → page).
    Validates that transformed objects are in the expected coordinate system.

    Example:
        # Table at position (200, 300) in page coordinates
        transform = CoordinateTransform(
            CoordinateSystem.PAGE_ABSOLUTE,
            CoordinateSystem.REGION_RELATIVE,
            offset_x=-200,
            offset_y=-300
        )

        # OCR box at (250, 350) in page coords → (50, 50) in table coords
        page_box = BBox(250, 350, 300, 380, CoordinateSystem.PAGE_ABSOLUTE)
        region_box = transform.transform_bbox(page_box)
        # region_box = BBox(50, 50, 100, 80, CoordinateSystem.REGION_RELATIVE)
    """
    from_system: CoordinateSystem
    to_system: CoordinateSystem
    offset_x: float
    offset_y: float

    def transform_bbox(self, bbox: BBox) -> BBox:
        """Transform a single bbox, validating coordinate systems."""
        if bbox.coord_system != self.from_system:
            raise ValueError(
                f"BBox is in {bbox.coord_system.value}, expected {self.from_system.value}"
            )

        return BBox(
            x_min=bbox.x_min + self.offset_x,
            y_min=bbox.y_min + self.offset_y,
            x_max=bbox.x_max + self.offset_x,
            y_max=bbox.y_max + self.offset_y,
            coord_system=self.to_system,
            label=bbox.label,
            confidence=bbox.confidence
        )

    def transform_list(self, bboxes: List[BBox]) -> List[BBox]:
        """Transform multiple bboxes."""
        return [self.transform_bbox(bbox) for bbox in bboxes]

    def transform_polygon(self, poly: BPolygon) -> BPolygon:
        """Transform a polygon."""
        if poly.coord_system != self.from_system:
            raise ValueError(
                f"Polygon is in {poly.coord_system.value}, expected {self.from_system.value}"
            )

        offset = np.array([self.offset_x, self.offset_y])
        transformed_points = poly.points + offset

        return BPolygon(
            points=transformed_points,
            coord_system=self.to_system,
            label=poly.label,
            confidence=poly.confidence
        )

    def inverse(self) -> 'CoordinateTransform':
        """Get reverse transformation (e.g., region → page if this is page → region)."""
        return CoordinateTransform(
            from_system=self.to_system,
            to_system=self.from_system,
            offset_x=-self.offset_x,
            offset_y=-self.offset_y
        )


class SpatialOps:
    """
    Single source of truth for all spatial operations.

    Replaces:
    - calculate_overlap_ratio() (layout_parsing/utils.py)
    - compute_iou() (table_recognition/pipeline_v2.py, table_recognition_post_processing_v2.py)
    - get_bbox_intersection() (layout_parsing/utils.py)
    - calculate_minimum_enclosing_bbox() (layout_parsing/utils.py)
    - update_region_box() (layout_parsing/utils.py)
    - get_overlap_boxes_idx() (layout_parsing/utils.py)

    All operations validate coordinate system compatibility.
    """

    @staticmethod
    def iou(bbox1: BBox, bbox2: BBox, mode: str = "union") -> float:
        """
        Calculate Intersection over Union (IoU) or variants.

        Args:
            bbox1, bbox2: Bounding boxes (must be in same coordinate system)
            mode:
                "union" - Standard IoU: intersection / (area1 + area2 - intersection)
                "small" - intersection / min(area1, area2)
                "large" - intersection / max(area1, area2)

        Returns:
            Overlap ratio in [0, 1]

        Raises:
            ValueError: If bboxes are in different coordinate systems
        """
        if bbox1.coord_system != bbox2.coord_system:
            raise ValueError(
                f"Coordinate system mismatch: {bbox1.coord_system.value} vs {bbox2.coord_system.value}"
            )

        # Calculate intersection
        x_min_inter = max(bbox1.x_min, bbox2.x_min)
        y_min_inter = max(bbox1.y_min, bbox2.y_min)
        x_max_inter = min(bbox1.x_max, bbox2.x_max)
        y_max_inter = min(bbox1.y_max, bbox2.y_max)

        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        inter_area = inter_width * inter_height

        # Calculate reference area based on mode
        if mode == "union":
            ref_area = bbox1.area + bbox2.area - inter_area
        elif mode == "small":
            ref_area = min(bbox1.area, bbox2.area)
        elif mode == "large":
            ref_area = max(bbox1.area, bbox2.area)
        else:
            raise ValueError(f"Invalid mode '{mode}', must be 'union', 'small', or 'large'")

        return inter_area / ref_area if ref_area > 0 else 0.0

    @staticmethod
    def intersection(bbox1: BBox, bbox2: BBox) -> Optional[BBox]:
        """
        Compute intersection bounding box.

        Returns:
            Intersection bbox, or None if no overlap
        """
        if bbox1.coord_system != bbox2.coord_system:
            raise ValueError("Coordinate system mismatch")

        x_min = max(bbox1.x_min, bbox2.x_min)
        y_min = max(bbox1.y_min, bbox2.y_min)
        x_max = min(bbox1.x_max, bbox2.x_max)
        y_max = min(bbox1.y_max, bbox2.y_max)

        if x_min >= x_max or y_min >= y_max:
            return None  # No intersection

        return BBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            coord_system=bbox1.coord_system
        )

    @staticmethod
    def union_bbox(bboxes: List[BBox]) -> BBox:
        """
        Calculate minimum enclosing bounding box.

        Replaces: calculate_minimum_enclosing_bbox()

        Args:
            bboxes: List of bounding boxes (must all be in same coordinate system)

        Returns:
            Smallest bbox containing all input bboxes
        """
        if not bboxes:
            raise ValueError("Empty bbox list")

        coord_sys = bboxes[0].coord_system
        if not all(b.coord_system == coord_sys for b in bboxes):
            raise ValueError("Mixed coordinate systems in bbox list")

        return BBox(
            x_min=min(b.x_min for b in bboxes),
            y_min=min(b.y_min for b in bboxes),
            x_max=max(b.x_max for b in bboxes),
            y_max=max(b.y_max for b in bboxes),
            coord_system=coord_sys
        )

    @staticmethod
    def expand_bbox(base: BBox, new: BBox) -> BBox:
        """
        Expand base bbox to include new bbox.

        Replaces: update_region_box()

        Returns:
            Union of the two bboxes
        """
        return SpatialOps.union_bbox([base, new])

    @staticmethod
    def filter_by_overlap(
        src_bboxes: List[BBox],
        ref_bbox: BBox,
        threshold: float = 0.0,
        mode: str = "small"
    ) -> List[int]:
        """
        Find indices of source bboxes overlapping with reference bbox.

        Replaces: get_overlap_boxes_idx() + manual filtering

        Args:
            src_bboxes: List of candidate bboxes to filter
            ref_bbox: Reference bbox to check overlap against
            threshold: Minimum IoU ratio to consider as match
            mode: IoU calculation mode (see iou() method)

        Returns:
            List of indices into src_bboxes that pass threshold
        """
        if not src_bboxes:
            return []

        # Validate all in same coordinate system
        if not all(b.coord_system == ref_bbox.coord_system for b in src_bboxes):
            raise ValueError("Mixed coordinate systems")

        indices = []
        for idx, bbox in enumerate(src_bboxes):
            if SpatialOps.iou(bbox, ref_bbox, mode) > threshold:
                indices.append(idx)

        return indices

    @staticmethod
    def projection_overlap(
        bbox1: BBox,
        bbox2: BBox,
        direction: str = "horizontal",
        mode: str = "union"
    ) -> float:
        """
        Calculate 1D overlap ratio along specified axis.

        Used for text line detection (group spans with high horizontal overlap).

        Args:
            bbox1, bbox2: Bounding boxes
            direction: "horizontal" (x-axis) or "vertical" (y-axis)
            mode: "union", "small", or "large"

        Returns:
            Overlap ratio in [0, 1]
        """
        if bbox1.coord_system != bbox2.coord_system:
            raise ValueError("Coordinate system mismatch")

        if direction == "horizontal":
            start1, end1 = bbox1.x_min, bbox1.x_max
            start2, end2 = bbox2.x_min, bbox2.x_max
        elif direction == "vertical":
            start1, end1 = bbox1.y_min, bbox1.y_max
            start2, end2 = bbox2.y_min, bbox2.y_max
        else:
            raise ValueError(f"Invalid direction '{direction}', must be 'horizontal' or 'vertical'")

        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap = max(0, overlap_end - overlap_start)

        # Calculate reference length
        if mode == "union":
            ref_length = max(end1, end2) - min(start1, start2)
        elif mode == "small":
            ref_length = min(end1 - start1, end2 - start2)
        elif mode == "large":
            ref_length = max(end1 - start1, end2 - start2)
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        return overlap / ref_length if ref_length > 0 else 0.0


class ConversionUtils:
    """
    Utilities for converting between legacy dict/list formats and new BBox/BPolygon.

    Use at pipeline boundaries:
    - Convert FROM model outputs (dicts/lists) TO BBox/BPolygon for internal processing
    - Convert FROM BBox/BPolygon TO dicts/lists for sub-pipeline calls
    """

    @staticmethod
    def bbox_from_dict(d: dict, coord_system: CoordinateSystem) -> BBox:
        """
        Convert DetResult box dict to BBox.

        Input format: {"coordinate": [x1,y1,x2,y2], "label": "...", "score": 0.95}
        """
        coord = d["coordinate"]
        return BBox(
            x_min=coord[0],
            y_min=coord[1],
            x_max=coord[2],
            y_max=coord[3],
            coord_system=coord_system,
            label=d.get("label"),
            confidence=d.get("score")
        )

    @staticmethod
    def bbox_from_list(
        coords: Union[List[float], np.ndarray],
        coord_system: CoordinateSystem,
        label: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> BBox:
        """
        Convert [x1,y1,x2,y2] list or array to BBox.
        """
        if isinstance(coords, np.ndarray):
            coords = coords.tolist()
        return BBox(
            x_min=coords[0],
            y_min=coords[1],
            x_max=coords[2],
            y_max=coords[3],
            coord_system=coord_system,
            label=label,
            confidence=confidence
        )

    @staticmethod
    def polygon_from_list(
        points: Union[List, np.ndarray],
        coord_system: CoordinateSystem,
        label: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> BPolygon:
        """
        Convert polygon points to BPolygon.

        Input formats:
        - List of tuples: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        - Nested list: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        - Numpy array: shape (N, 2)
        """
        points_array = np.array(points).reshape(-1, 2)
        return BPolygon(
            points=points_array,
            coord_system=coord_system,
            label=label,
            confidence=confidence
        )

    @staticmethod
    def bboxes_from_det_result(det_result: dict, coord_system: CoordinateSystem) -> List[BBox]:
        """
        Convert DetResult["boxes"] to List[BBox].

        DetResult format:
        {
            "boxes": [
                {"coordinate": [x1,y1,x2,y2], "label": "text", "score": 0.95},
                {"coordinate": [x1,y1,x2,y2], "label": "table", "score": 0.88},
                ...
            ]
        }
        """
        return [
            ConversionUtils.bbox_from_dict(box, coord_system)
            for box in det_result.get("boxes", [])
        ]

    @staticmethod
    def bboxes_from_ocr_result(ocr_result: dict, coord_system: CoordinateSystem) -> List[BBox]:
        """
        Convert OCRResult["rec_boxes"] to List[BBox].

        OCRResult format:
        {
            "rec_boxes": np.ndarray shape (N, 4),  # [x1,y1,x2,y2] rows
            "rec_texts": ["text1", "text2", ...],
            "rec_labels": ["text", "text", ...],
            "rec_scores": [0.95, 0.88, ...]
        }
        """
        rec_boxes = ocr_result.get("rec_boxes", np.array([]))
        if len(rec_boxes) == 0:
            return []

        # Handle case where rec_boxes is numpy array or list
        if isinstance(rec_boxes, np.ndarray) and len(rec_boxes.shape) == 1:
            rec_boxes = rec_boxes.reshape(1, -1)

        rec_labels = ocr_result.get("rec_labels", [None] * len(rec_boxes))
        rec_scores = ocr_result.get("rec_scores", [None] * len(rec_boxes))

        bboxes = []
        for idx, box in enumerate(rec_boxes):
            label = rec_labels[idx] if idx < len(rec_labels) else None
            confidence = rec_scores[idx] if idx < len(rec_scores) else None
            bboxes.append(
                ConversionUtils.bbox_from_list(box, coord_system, label, confidence)
            )

        return bboxes

    @staticmethod
    def det_result_from_bboxes(bboxes: List[BBox]) -> dict:
        """
        Convert List[BBox] back to DetResult format.

        Used when calling sub-pipelines that expect legacy format.
        """
        return {
            "boxes": [bbox.to_dict() for bbox in bboxes]
        }
