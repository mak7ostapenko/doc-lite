"""Base exporter abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.ocr_core.result import GeneralOCRResultV2



class BaseExporter(ABC):
    """
    Abstract base class for all result exporters.

    Each exporter:
    - Takes GeneralOCRResultV2 as input
    - Produces specific output format
    - Maintains 100% backward compatibility
    - Is independently testable
    """

    @abstractmethod
    def export(self, result: 'GeneralOCRResultV2', output_path: Path) -> Any:
        """
        Export result to specified format.

        Args:
            result: GeneralOCRResultV2 instance to export
            output_path: Directory or file path for output

        Returns:
            Exported data (format depends on exporter type)
        """
        pass
