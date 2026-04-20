import argparse
import os
import cv2
from src.pp.pipelines import create_pipeline
from src.pp.pipelines.base import BasePipeline

PIPELINE_CONFIGS = {
    "full":     "PP-StructureV3",
    "lite":     "configs/pipelines/PP-StructureV3-lite.yaml",
    "ocr-only": "configs/pipelines/PP-StructureV3-lite.yaml",
}


def process_lang_images(
    test_images: list,
    pipeline: BasePipeline,
    lang: str,
    result_dir: str,
    iqa_model=None,
    use_region_detection: bool = True,
):
    for img_path in test_images:

        if iqa_model is not None:
            img = cv2.imread(img_path)
            image_score = iqa_model.predict(img_bgr=img)
            score_label = 2 # TODO: we need to benchmark to find thresholds
        else:
            image_score = None
            score_label = None

        # TODO: depending on the score to processing

        try:
            output = pipeline.predict(
                img_path,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
                use_seal_recognition=False,
                use_table_recognition=False,
                use_formula_recognition=False,
                use_region_detection=use_region_detection,
                lang=lang,
                # Table Recognition Mode (choose one):
                use_e2e_wired_table_rec_model=False,      # End-to-end wired (bordered) tables
                use_e2e_wireless_table_rec_model=False,   # End-to-end wireless (borderless) tables

                # Table Processing Options:
                use_wired_table_cells_trans_to_html=False,       # Wired cells → HTML
                use_wireless_table_cells_trans_to_html=False,    # Wireless cells → HTML
                use_table_orientation_classify=False,              # Auto-rotate tables
                use_ocr_results_with_table_cells=False,           # Merge OCR with cell detection

                # OCR Parameters (affect table text extraction):
                text_det_limit_side_len=960,    # Image resize limit (default: 960) 1216
                text_det_limit_type=None,        # "max" or "min" 
                text_det_thresh=0.3,            # Detection threshold (default: 0.3)
                text_det_box_thresh=0.6,        # Box threshold (default: 0.6)
                text_det_unclip_ratio=2.0,      # Unclip ratio (default: 2.0)
                text_rec_score_thresh=0,      # Recognition confidence (default: 0)
            )

            output_dir = os.path.join(result_dir, f"{os.path.basename(img_path)}/")
            os.makedirs(output_dir, exist_ok=True)
            for res in output:
                # res.print()
                res.save_to_img(output_dir)
                res.save_to_json(output_dir)
                res.save_to_xlsx(output_dir)
                res.save_to_markdown(output_dir)

        except Exception as e:
            print(f"✗ Error processing {img_path}:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


def main(mode: str = "lite"):
    data_root = "data/some_test_data/"
    result_root = f"out/results-{mode}/"

    eng_images_dir = os.path.join(data_root, "en/")
    uk_images_dir = os.path.join(data_root, "uk/")

    test_data = [
        ('uk', uk_images_dir),
        ('en', eng_images_dir),
    ]

    print("=" * 80)
    print("Flat PP-StructureV3 Pipeline - Model Initialization")
    print("=" * 80)

    ####################################################
    # 1. INITIALISATION STEP
    ####################################################

    # IMAGE QUALITY ASSESSMENT
    if USE_IMG_QUALITY_ASSESMENT:
        from src.image_quality.model import  ImgQualityAssessmentModel 
        iqa_model = ImgQualityAssessmentModel(model_name='musiq')
    else:
        iqa_model = None

    # OCR MODELS
    ocr_pipeline: BasePipeline = create_pipeline(pipeline=PIPELINE_CONFIGS[mode])

    ####################################################
    # 2. PROCESSING STEP
    ####################################################

    os.makedirs(result_root, exist_ok=True)
    for lang, imgs_dir in test_data:
        print(f"\n{'-' * 30}\nLanguage: {lang}\n{'-' * 30}")

        if os.path.isdir(imgs_dir):
            test_images = [
                os.path.join(imgs_dir, name) 
                for name in os.listdir(imgs_dir) 
            ]
        else:
            test_images = [imgs_dir]

        result_dir = os.path.join(result_root, lang)
        os.makedirs(result_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("Processing Images")
        print("=" * 80) 

        if lang in ['ru', 'uk']:
            lang = 'cyrillic'   # needed for model selection
            
        process_lang_images(
            test_images=test_images,
            pipeline=ocr_pipeline,
            lang=lang,
            result_dir=result_dir,
            iqa_model=iqa_model,
            use_region_detection=(mode != "ocr-only"),
        )

        print("\n" + "=" * 80)
        print("Pipeline Complete")
        print("=" * 80)
        print(f"\nResults saved to: {result_root}")

    print()    
    print("\nNext steps:")
    print("  1. Check output_flat/ for results")
    print("  2. Compare with output_orig/ to verify correctness")
    print("  3. Run: pytest tests/ -v for full test suite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "lite", "ocr-only"],
        default="lite",
        help="full: all models (~3-4 GB RAM); lite: no table/formula (~1 GB); ocr-only: OCR text only (~600 MB)",
    )
    args = parser.parse_args()
    USE_IMG_QUALITY_ASSESMENT = False
    main(mode=args.mode)