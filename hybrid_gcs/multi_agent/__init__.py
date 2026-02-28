"""
Multi-Agent Module for Hybrid-GCS.

Implements multi-agent extensions including:
- Communication: Attention-based MARL communication (CTDE)
- ST-GCS: Space-Time Graphs of Convex Sets for collision avoidance

References:
    Section 4 of Hybrid-GCS Theory.
"""

from .communication import AttentionComm, CentralizedCritic, MultiAgentPolicy
from .st_gcs import SpaceTimeEdge, SpaceTimeGCS, SpaceTimeVertex

__all__ = [
    "AttentionComm",
    "MultiAgentPolicy",
    "CentralizedCritic",
    "SpaceTimeVertex",
    "SpaceTimeEdge",
    "SpaceTimeGCS",
]
