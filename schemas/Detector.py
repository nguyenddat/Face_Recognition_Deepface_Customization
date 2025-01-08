from abc import ABC, abstractmethod
from typing import *
from dataclasses import dataclass 
import numpy as np

class Detector(ABC):
    @abstractmethod
    def detect_faces(self, img: np.ndarray) -> List["FacialAreaRegion"]:
        pass

@dataclass
class FacialAreaRegion:
    x: int; y: int; w: int; h: int
    left_eye: Optional[Tuple[int, int]] = None 
    right_eye: Optional[Tuple[int, int]] = None
    nose: Optional[Tuple[int, int]] = None
    left_mouth: Optional[Tuple[int, int]] = None
    right_mouth: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None

@dataclass 
class DetectedFace:
    img: np.ndarray
    facial_area: FacialAreaRegion
    confidence: float