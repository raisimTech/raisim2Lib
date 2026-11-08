from typing import overload

import numpy


@overload
def clamp01(value: float) -> float | numpy.ndarray:
    """C++: float (float)"""

@overload
def clamp01(color: numpy.ndarray) -> float | numpy.ndarray:
    """C++: glm::vec4 (const glm::vec4 &)"""

def clampRgb255(color: numpy.ndarray) -> numpy.ndarray:
    """C++: glm::vec4 (const glm::vec4 &)"""

def rgba255ToUnit(color: numpy.ndarray) -> numpy.ndarray:
    """C++: glm::vec4 (const glm::vec4 &)"""

def unitToRgba255(color: numpy.ndarray) -> numpy.ndarray:
    """C++: glm::vec4 (const glm::vec4 &)"""

def applyDisplayGamma(linearColor: numpy.ndarray, gamma: float) -> numpy.ndarray:
    """C++: glm::vec4 (const glm::vec4 &, float)"""
