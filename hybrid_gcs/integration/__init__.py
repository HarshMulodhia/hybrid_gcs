"""
Integration Module for Hybrid-GCS.

Implements hybrid GCS + RL integration including action blending,
safety filtering, and dual-pathway feature extraction.

Modules:
    - blending: Action blending methods (weighted, hierarchical, conflict resolution)
    - safety_filter: Real-time constraint enforcement
    - feature_extractor: Dual pathway feature extraction for GCS and RL
"""

from .blending import (
    BlendingMethod,
    ConflictResolutionBlender,
    HierarchicalBlender,
    PriorityNetworkBlender,
    WeightedBlender,
)
from .feature_extractor import DualPathwayExtractor, FeatureExtractorConfig
from .safety_filter import ControlBarrierFilter, SafetyFilter, SafetyFilterConfig

__all__ = [
    "BlendingMethod",
    "WeightedBlender",
    "HierarchicalBlender",
    "ConflictResolutionBlender",
    "PriorityNetworkBlender",
    "SafetyFilter",
    "SafetyFilterConfig",
    "ControlBarrierFilter",
    "DualPathwayExtractor",
    "FeatureExtractorConfig",
]
