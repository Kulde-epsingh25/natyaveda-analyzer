"""
NatyaVeda — Person Tracker
Wraps ByteTrack-style tracking to follow the principal dancer across frames.
Used inside DanceIsolator to maintain consistent dancer identity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]    # x1,y1,x2,y2
    confidence: float
    age: int = 0
    hits: int = 0
    lost: int = 0
    history: list = field(default_factory=list)

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class SimpleTracker:
    """
    Lightweight IoU-based tracker as ByteTrack fallback.
    Assigns consistent IDs to detections across frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(
        self, detections: list[tuple[int, int, int, int, float]]
    ) -> list[Track]:
        """
        Update tracker with new detections.
        detections: list of (x1, y1, x2, y2, confidence)
        Returns: list of active Track objects
        """
        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(
                    track_id=self._next_id,
                    bbox=det[:4], confidence=det[4]
                ))
                self._next_id += 1
            return [t for t in self.tracks if t.lost == 0]

        # Compute IoU matrix
        iou_mat = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_mat[i, j] = self._iou(track.bbox, det[:4])

        # Greedy matching
        matched_tracks = set()
        matched_dets   = set()

        while True:
            if iou_mat.size == 0:
                break
            max_idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            if iou_mat[max_idx] < self.iou_threshold:
                break
            ti, di = max_idx
            if ti in matched_tracks or di in matched_dets:
                iou_mat[ti, di] = 0
                continue
            self.tracks[ti].bbox       = detections[di][:4]
            self.tracks[ti].confidence = detections[di][4]
            self.tracks[ti].hits      += 1
            self.tracks[ti].lost       = 0
            self.tracks[ti].age       += 1
            matched_tracks.add(ti)
            matched_dets.add(di)
            iou_mat[ti, :] = 0
            iou_mat[:, di] = 0

        # Increment lost for unmatched tracks
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.lost += 1

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(
                    track_id=self._next_id,
                    bbox=det[:4], confidence=det[4]
                ))
                self._next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        return [t for t in self.tracks if t.lost == 0]

    @staticmethod
    def _iou(boxA: tuple, boxB: tuple) -> float:
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(areaA + areaB - inter + 1e-8)

    def get_principal_dancer(self, strategy: str = "largest") -> Optional[Track]:
        """Return the track most likely to be the principal dancer."""
        active = [t for t in self.tracks if t.lost == 0]
        if not active:
            return None
        if strategy == "largest":
            return max(active, key=lambda t: t.area)
        if strategy == "most_hits":
            return max(active, key=lambda t: t.hits)
        return active[0]
