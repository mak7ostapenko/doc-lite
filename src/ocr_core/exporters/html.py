"""
HTMLExporter - Exports table results to HTML format.

Maintains 100% backward compatibility with original _to_html() method.
"""

from pathlib import Path
from typing import Dict

from .base import BaseExporter

from src.ocr_core.result import GeneralOCRResultV2


class HTMLExporter(BaseExporter):
    """
    Export table results as HTML.

    Exact copy of logic from original _to_html() method (lines 295-309).
    """

    def export(self, result: 'GeneralOCRResultV2', output_path: Path) -> Dict[str, str]:
        """
        Generate HTML representation of tables.

        Args:
            result: GeneralOCRResultV2 instance
            output_path: Directory to save HTML files

        Returns:
            Dictionary mapping table IDs to HTML strings
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model_settings = result.model_settings
        res_html_dict = {}

        if model_settings.get("use_table_recognition", False) and len(result.tables) > 0:
            for sno in range(len(result.tables)):
                table_res = result.tables[sno]
                table_region_id = table_res.get("table_region_id")
                key = f"table_{table_region_id}"

                if hasattr(table_res, 'html'):
                    res_html_dict[key] = table_res.html["pred"]
                elif isinstance(table_res, dict) and "pred_html" in table_res:
                    res_html_dict[key] = table_res["pred_html"]

                # Save to file
                if key in res_html_dict:
                    html_file = output_path / f"{key}.html"
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(res_html_dict[key])

        return res_html_dict
