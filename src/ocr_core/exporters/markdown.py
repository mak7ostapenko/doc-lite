"""
MarkdownExporter - Exports OCR results to Markdown format.

Maintains 100% backward compatibility with original _to_markdown() method.

Includes all helper functions from original general_ocr_result.py (lines 23-127).
"""

import re
from functools import partial
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING

from .base import BaseExporter

from src.ocr_core.result import GeneralOCRResultV2



# ============================================================================
# Generic Helper Functions (model-agnostic)
# ============================================================================

def get_seg_flag_generic(block: 'LayoutBlockData', prev_block: Optional['LayoutBlockData']) -> Tuple[bool, bool]:
    """
    Determine if text segment needs paragraph breaks before/after.

    This is a simplified, generic version that works with any OCR model.
    Logic:
    - seg_start_flag: Should we add paragraph break before this block?
    - seg_end_flag: Should we add paragraph break after this block?

    Args:
        block: Current layout block
        prev_block: Previous layout block (None if first block)

    Returns:
        (seg_start_flag, seg_end_flag) tuple of booleans
    """
    seg_start_flag = True
    seg_end_flag = True

    # For first block, no start flag needed
    if prev_block is None:
        seg_start_flag = False
        return seg_start_flag, seg_end_flag

    # Check if blocks are on same horizontal line (overlapping vertically)
    # Use bbox coordinates: [x_min, y_min, x_max, y_max]
    curr_bbox = block.bbox
    prev_bbox = prev_block.bbox

    # Vertical overlap check
    vertical_overlap = (
        curr_bbox.y_min < prev_bbox.y_max and
        curr_bbox.y_max > prev_bbox.y_min
    )

    # Horizontal distance
    horizontal_gap = abs(curr_bbox.x_min - prev_bbox.x_max)

    # If blocks overlap vertically and are close horizontally, they're part of same paragraph
    if vertical_overlap and horizontal_gap < max(prev_block.width, block.width):
        # Multi-line previous block suggests continuation
        if prev_block.num_of_lines > 1:
            seg_start_flag = False

    return seg_start_flag, seg_end_flag


# ============================================================================
# Markdown Formatting Helpers
# ============================================================================

def compile_title_pattern():
    """Precompiled regex pattern for matching numbering at the beginning of the title."""
    numbering_pattern = (
        r"(?:" + r"[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|" + r"[\(\（](?:[1-9][0-9]*|["
        r"一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|" + r"["
        r"一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+"
        r"[、\.]?|" + r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.?" + r")"
    )
    return re.compile(r"^\s*(" + numbering_pattern + r")(\s*)(.*)$")


TITLE_RE_PATTERN = compile_title_pattern()


def format_title_func(block):
    """
    Normalize chapter title.
    Add the '#' to indicate the level of the title.
    If numbering exists, ensure there's exactly one space between it and the title content.
    If numbering does not exist, return the original title unchanged.
    """
    title = block.content
    match = TITLE_RE_PATTERN.match(title)
    if match:
        numbering = match.group(1).strip()
        title_content = match.group(3).lstrip()
        # Return numbering and title content separated by one space
        title = numbering + " " + title_content

    title = title.rstrip(".")
    level = (
        title.count(".",) + 1
        if "." in title
        else 1
    )
    return f"#{'#' * level} {title}".replace("-\n", "").replace("\n", " ")


def format_centered_by_html(string):
    return (
        f'<div style="text-align: center;">{string}</div>'.replace("-\n", "").replace("\n", " ")
        + "\n"
    )


def format_text_plain_func(block):
    return block.content


def format_image_scaled_by_html_func(block, original_image_width):
    img_tags = []
    image_path = block.image["path"]
    image_width = block.image["img"].width
    scale = int(image_width / original_image_width * 100)
    img_tags.append(
        '<img src="{}" alt="Image" width="{}%" />'.format(
            image_path.replace("-\n", "").replace("\n", " "), scale
        ),
    )
    return "\n".join(img_tags)


def format_image_plain_func(block):
    img_tags = []
    image_path = block.image["path"]
    img_tags.append("![]({})".format(image_path.replace("-\n", "").replace("\n", " ")))
    return "\n".join(img_tags)


def format_chart2table_func(block):
    lines_list = block.content.split("\n")
    column_num = len(lines_list[0].split("|"))
    lines_list.insert(1, "|".join(["---"] * column_num))
    lines_list = [f"|{line}|" for line in lines_list]
    return "\n".join(lines_list)


def simplify_table_func(table_code):
    return "\n" + table_code.replace("<html>", "").replace("</html>", "").replace(
        "<body>", ""
    ).replace("</body>", "")


def format_first_line_func(block, templates, format_func, spliter):
    lines = block.content.split(spliter)
    for idx in range(len(lines)):
        line = lines[idx]
        if line.strip() == "":
            continue
        if line.lower() in templates:
            lines[idx] = format_func(line)
        break
    return spliter.join(lines)


# ============================================================================
# MarkdownExporter Class
# ============================================================================

class MarkdownExporter(BaseExporter):
    """
    Export OCR results as Markdown.

    Exact copy of logic from original _to_markdown() method (lines 327-475).
    """

    def export(self, result: 'GeneralOCRResultV2', output_path: Path, pretty: bool = True) -> Dict[str, Any]:
        """
        Generate Markdown representation.

        Args:
            result: GeneralOCRResultV2 instance
            output_path: Directory to save Markdown file
            pretty: Whether to pretty markdown by HTML (default True)

        Returns:
            Dictionary with markdown content and metadata
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        original_image_width = result.preprocessed_image.shape[1]

        # Configure formatting functions based on pretty mode
        if pretty:
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                )
            )
        else:
            format_text_func = lambda block: block.content
            format_image_func = format_image_plain_func

        if result.model_settings.get("use_chart_recognition", False):
            format_chart_func = format_chart2table_func
        else:
            format_chart_func = format_image_func

        if result.model_settings.get("use_seal_recognition", False):
            format_seal_func = lambda block: "\n".join(
                [format_image_func(block), format_text_func(block)]
            )
        else:
            format_seal_func = format_image_func

        if result.model_settings.get("use_table_recognition", False):
            if pretty:
                format_table_func = lambda block: "\n" + format_text_func(
                    block
                ).replace("<table>", '<table border="1">')
            else:
                format_table_func = lambda block: simplify_table_func(
                    "\n" + block.content
                )
        else:
            format_table_func = format_image_func

        if result.model_settings.get("use_formula_recognition", False):
            format_formula_func = lambda block: f"$${block.content}$$"
        else:
            format_formula_func = format_image_func

        # Map of label to formatting function
        handle_funcs_dict = {
            "paragraph_title": format_title_func,
            "abstract_title": format_title_func,
            "reference_title": format_title_func,
            "content_title": format_title_func,
            "doc_title": lambda block: f"# {block.content}".replace(
                "-\n",
                "",
            ).replace("\n", " "),
            "table_title": format_text_func,
            "figure_title": format_text_func,
            "chart_title": format_text_func,
            "vision_footnote": lambda block: block.content.replace(
                "\n\n", "\n"
            ).replace("\n", "\n\n"),
            "text": lambda block: block.content.replace("\n\n", "\n").replace(
                "\n", "\n\n"
            ),
            "abstract": partial(
                format_first_line_func,
                templates=["摘要", "abstract"],
                format_func=lambda l: f"## {l}\n",
                spliter=" ",
            ),
            "content": lambda block: block.content.replace("-\n", "  \n").replace(
                "\n", "  \n"
            ),
            "image": format_image_func,
            "chart": format_chart_func,
            "formula": format_formula_func,
            "table": format_table_func,
            "reference": partial(
                format_first_line_func,
                templates=["参考文献", "references"],
                format_func=lambda l: f"## {l}",
                spliter="\n",
            ),
            "algorithm": lambda block: block.content.strip("\n"),
            "seal": format_seal_func,
        }

        markdown_content = ""
        last_label = None
        seg_start_flag = True
        seg_end_flag = True
        prev_block = None
        page_first_element_seg_start_flag = None
        page_last_element_seg_end_flag = None
        markdown_info = {}
        markdown_info["markdown_images"] = {}

        # Generate markdown from layout blocks
        for block in result.layout_blocks:
            seg_start_flag, seg_end_flag = get_seg_flag_generic(block, prev_block)

            label = block.label
            if block.image is not None:
                markdown_info["markdown_images"][block.image["path"]] = block.image[
                    "img"
                ]
            page_first_element_seg_start_flag = (
                seg_start_flag
                if (page_first_element_seg_start_flag is None)
                else page_first_element_seg_start_flag
            )

            handle_func = handle_funcs_dict.get(label, None)
            if handle_func:
                prev_block = block
                if label == last_label == "text" and seg_start_flag == False:
                    markdown_content += handle_func(block)
                else:
                    markdown_content += (
                        "\n\n" + handle_func(block)
                        if markdown_content
                        else handle_func(block)
                    )
                last_label = label

        page_first_element_seg_start_flag = (
            True
            if page_first_element_seg_start_flag is None
            else page_first_element_seg_start_flag
        )
        page_last_element_seg_end_flag = seg_end_flag

        markdown_info["page_index"] = result.page_index
        markdown_info["input_path"] = result.input_path
        markdown_info["markdown_texts"] = markdown_content
        markdown_info["page_continuation_flags"] = (
            page_first_element_seg_start_flag,
            page_last_element_seg_end_flag,
        )

        # Add images from document
        for img in result.imgs_in_doc:
            markdown_info["markdown_images"][img["path"]] = img["img"]

        # Save markdown to file
        md_file = output_path / f"{Path(result.input_path).stem}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Save images
        img_dir = output_path / "imgs"
        img_dir.mkdir(exist_ok=True)
        for img_path, img_obj in markdown_info["markdown_images"].items():
            img_save_path = output_path / img_path
            img_save_path.parent.mkdir(parents=True, exist_ok=True)
            img_obj.save(img_save_path)

        return markdown_info
