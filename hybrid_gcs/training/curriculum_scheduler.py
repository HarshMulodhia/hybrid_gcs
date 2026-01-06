"""
Curriculum Learning Scheduler for Hybrid-GCS.

Implements progressive difficulty scheduling for training.
Supports multiple curriculum strategies:
- Linear progression
- Exponential progression
- Step-based progression
- Performance-based progression
"""

from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, List


class CurriculumType(Enum):
    """Types of curriculum learning schedules."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    PERFORMANCE = "performance"
    SIGMOID = "sigmoid"


class CurriculumSchedule(ABC):
    """Base class for curriculum schedules."""
    
    def __init__(self, schedule_type: CurriculumType):
        """
        Initialize curriculum schedule.
        
        Args:
            schedule_type: Type of curriculum
        """
        self.schedule_type = schedule_type
    
    @abstractmethod
    def get_difficulty(self, step: int) -> float:
        """
        Get difficulty for given step.
        
        Args:
            step: Current training step
        
        Returns:
            Difficulty in range [0, 1]
        """
        pass


class LinearCurriculum(CurriculumSchedule):
    """
    Linear progression from initial to final difficulty.
    
    difficulty(t) = initial + (final - initial) * t / total_steps
    """
    
    def __init__(self, initial_difficulty: float = 0.0, final_difficulty: float = 1.0,
                 total_steps: int = 10000):
        """
        Initialize linear curriculum.
        
        Args:
            initial_difficulty: Starting difficulty [0, 1]
            final_difficulty: Target difficulty [0, 1]
            total_steps: Number of steps to reach final difficulty
        """
        super().__init__(CurriculumType.LINEAR)
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.total_steps = total_steps
    
    def get_difficulty(self, step: int) -> float:
        """Get linear interpolated difficulty."""
        progress = min(step / self.total_steps, 1.0)
        difficulty = (
            self.initial_difficulty +
            (self.final_difficulty - self.initial_difficulty) * progress
        )
        return float(difficulty)


class ExponentialCurriculum(CurriculumSchedule):
    """
    Exponential progression - faster at beginning, slower at end.
    
    difficulty(t) = initial + (final - initial) * (1 - exp(-k*t))
    """
    
    def __init__(self, initial_difficulty: float = 0.0, final_difficulty: float = 1.0,
                 total_steps: int = 10000, rate: float = 0.01):
        """
        Initialize exponential curriculum.
        
        Args:
            initial_difficulty: Starting difficulty
            final_difficulty: Target difficulty
            total_steps: Reference number of steps
            rate: Exponential growth rate
        """
        super().__init__(CurriculumType.EXPONENTIAL)
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.total_steps = total_steps
        self.rate = rate
    
    def get_difficulty(self, step: int) -> float:
        """Get exponential difficulty."""
        progress = 1.0 - np.exp(-self.rate * step / self.total_steps)
        progress = min(progress, 1.0)
        difficulty = (
            self.initial_difficulty +
            (self.final_difficulty - self.initial_difficulty) * progress
        )
        return float(difficulty)


class StepCurriculum(CurriculumSchedule):
    """
    Step-wise progression with discrete difficulty levels.
    
    Difficulty increases in steps at predefined milestones.
    """
    
    def __init__(self, step_sizes: List[int] = None, difficulties: List[float] = None):
        """
        Initialize step curriculum.
        
        Args:
            step_sizes: List of step milestones (e.g., [1000, 5000, 10000])
            difficulties: List of difficulties at milestones (e.g., [0.3, 0.6, 1.0])
        """
        super().__init__(CurriculumType.STEP)
        
        if step_sizes is None:
            step_sizes = [1000, 5000, 10000]
        if difficulties is None:
            difficulties = [0.3, 0.6, 1.0]
        
        assert len(step_sizes) == len(difficulties), \
            "step_sizes and difficulties must have same length"
        
        self.step_sizes = step_sizes
        self.difficulties = difficulties
    
    def get_difficulty(self, step: int) -> float:
        """Get step-wise difficulty."""
        for i, milestone in enumerate(self.step_sizes):
            if step < milestone:
                return self.difficulties[i]
        return self.difficulties[-1]


class SigmoidCurriculum(CurriculumSchedule):
    """
    Sigmoid progression - slow start, fast middle, slow end.
    
    Creates S-shaped difficulty curve.
    """
    
    def __init__(self, initial_difficulty: float = 0.0, final_difficulty: float = 1.0,
                 total_steps: int = 10000, midpoint: float = 0.5, steepness: float = 5.0):
        """
        Initialize sigmoid curriculum.
        
        Args:
            initial_difficulty: Starting difficulty
            final_difficulty: Target difficulty
            total_steps: Number of steps for progression
            midpoint: Where S-curve inflection occurs (0-1)
            steepness: Steepness of S-curve
        """
        super().__init__(CurriculumType.SIGMOID)
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.total_steps = total_steps
        self.midpoint = midpoint
        self.steepness = steepness
    
    def get_difficulty(self, step: int) -> float:
        """Get sigmoid difficulty."""
        progress = step / self.total_steps
        sigmoid_val = 1.0 / (1.0 + np.exp(-self.steepness * (progress - self.midpoint)))
        difficulty = (
            self.initial_difficulty +
            (self.final_difficulty - self.initial_difficulty) * sigmoid_val
        )
        return float(difficulty)


class PerformanceCurriculum(CurriculumSchedule):
    """
    Performance-based curriculum that adapts to agent performance.
    
    Increases difficulty when performance is good.
    Decreases difficulty when performance is poor.
    """
    
    def __init__(self, initial_difficulty: float = 0.1, max_difficulty: float = 1.0,
                 success_threshold: float = 0.8, failure_threshold: float = 0.3,
                 increase_rate: float = 0.05, decrease_rate: float = 0.02):
        """
        Initialize performance-based curriculum.
        
        Args:
            initial_difficulty: Starting difficulty
            max_difficulty: Maximum allowed difficulty
            success_threshold: Performance threshold to increase difficulty
            failure_threshold: Performance threshold to decrease difficulty
            increase_rate: How much to increase per success
            decrease_rate: How much to decrease per failure
        """
        super().__init__(CurriculumType.PERFORMANCE)
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
    
    def get_difficulty(self, step: int) -> float:
        """Get current difficulty."""
        return self.current_difficulty
    
    def update(self, performance: float):
        """
        Update difficulty based on performance.
        
        Args:
            performance: Agent performance metric in [0, 1]
        """
        if performance >= self.success_threshold:
            # Increase difficulty
            self.current_difficulty = min(
                self.current_difficulty + self.increase_rate,
                self.max_difficulty
            )
        elif performance < self.failure_threshold:
            # Decrease difficulty
            self.current_difficulty = max(
                self.current_difficulty - self.decrease_rate,
                0.0
            )


@dataclass
class CurriculumConfig:
    """Configuration for curriculum manager."""
    
    schedule_type: CurriculumType = CurriculumType.LINEAR
    initial_difficulty: float = 0.0
    final_difficulty: float = 1.0
    total_steps: int = 10000
    
    # Linear/Exponential parameters
    rate: float = 0.01
    
    # Step curriculum parameters
    step_sizes: Optional[List[int]] = None
    difficulties: Optional[List[float]] = None
    
    # Sigmoid parameters
    midpoint: float = 0.5
    steepness: float = 5.0
    
    # Performance-based parameters
    success_threshold: float = 0.8
    failure_threshold: float = 0.3
    increase_rate: float = 0.05
    decrease_rate: float = 0.02


class CurriculumManager:
    """
    Manages curriculum learning.
    
    Tracks progress and provides difficulty scheduling for training.
    """
    
    def __init__(self, schedule: CurriculumSchedule = None,
                 config: CurriculumConfig = None):
        """
        Initialize curriculum manager.
        
        Args:
            schedule: CurriculumSchedule instance
            config: CurriculumConfig for creating schedule
        """
        if schedule is not None:
            self.schedule = schedule
        elif config is not None:
            self.schedule = self._create_schedule(config)
        else:
            # Default: linear curriculum
            self.schedule = LinearCurriculum()
        
        self.current_step = 0
        self.history: Dict[int, float] = {}
    
    def _create_schedule(self, config: CurriculumConfig) -> CurriculumSchedule:
        """Create schedule from configuration."""
        if config.schedule_type == CurriculumType.LINEAR:
            return LinearCurriculum(
                config.initial_difficulty,
                config.final_difficulty,
                config.total_steps
            )
        elif config.schedule_type == CurriculumType.EXPONENTIAL:
            return ExponentialCurriculum(
                config.initial_difficulty,
                config.final_difficulty,
                config.total_steps,
                config.rate
            )
        elif config.schedule_type == CurriculumType.STEP:
            return StepCurriculum(config.step_sizes, config.difficulties)
        elif config.schedule_type == CurriculumType.SIGMOID:
            return SigmoidCurriculum(
                config.initial_difficulty,
                config.final_difficulty,
                config.total_steps,
                config.midpoint,
                config.steepness
            )
        elif config.schedule_type == CurriculumType.PERFORMANCE:
            return PerformanceCurriculum(
                config.initial_difficulty,
                config.final_difficulty,
                config.success_threshold,
                config.failure_threshold,
                config.increase_rate,
                config.decrease_rate
            )
        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")
    
    def get_current_difficulty(self, step: int = None) -> float:
        """
        Get difficulty for current or specified step.
        
        Args:
            step: Training step (uses current_step if None)
        
        Returns:
            Difficulty in [0, 1]
        """
        if step is None:
            step = self.current_step
        
        difficulty = self.schedule.get_difficulty(step)
        self.history[step] = difficulty
        return difficulty
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1
    
    def update_performance(self, performance: float):
        """
        Update based on performance (for performance-based curriculum).
        
        Args:
            performance: Performance metric [0, 1]
        """
        if isinstance(self.schedule, PerformanceCurriculum):
            self.schedule.update(performance)
    
    def get_history(self) -> Dict[int, float]:
        """Get difficulty history."""
        return self.history.copy()
    
    def reset(self):
        """Reset curriculum manager."""
        self.current_step = 0
        self.history = {}
    
    def __repr__(self) -> str:
        """String representation."""
        difficulty = self.get_current_difficulty()
        return (
            f"CurriculumManager("
            f"schedule={self.schedule.schedule_type.value}, "
            f"step={self.current_step}, "
            f"difficulty={difficulty:.3f})"
        )
