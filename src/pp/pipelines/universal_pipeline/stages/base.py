"""
Base class for pipeline stages.

Stages are composable units that perform specific operations in the pipeline.
Each stage reads from the context, performs its operation, and writes back to the context.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.pp.pipelines.universal_pipeline.context import PipelineContext


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    A stage is a single unit of work in the pipeline that:
    1. Reads inputs from PipelineContext
    2. Performs a specific operation
    3. Writes outputs back to PipelineContext

    Stages can be composed to create flexible pipelines.
    """

    def __init__(self, pipeline: Any):
        """
        Initialize stage with reference to parent pipeline.

        Args:
            pipeline: The parent _LayoutParsingPipelineV2 instance.
                     Provides access to models and configuration.
        """
        self.pipeline = pipeline

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the stage operation.

        Args:
            context: Pipeline context with inputs

        Returns:
            Updated context with outputs populated
        """
        pass

    @property
    def name(self) -> str:
        """Get stage name for logging and debugging."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"
