# *************************************************************************
# GScore: LLM-as-judge visual quality scoring, converted from GScore.ipynb.
#
# ASSUMPTION / OPEN QUESTION (please confirm or correct):
# The original notebook used the Vertex AI SDK (`vertexai` package), which requires a
# GCP project ID + region + a service-account credentials file — not just a single API
# key. Since the plan was to store credentials in a single `.env` file, this script
# instead uses the simpler **Gemini Developer API** (Google AI Studio), authenticated
# with one `GEMINI_API_KEY` value. If you actually need Vertex AI (e.g. an existing
# GCP project/billing setup), this file's `_load_gemini_client()` / `compute_gscore()`
# would need to be swapped for the `vertexai` SDK instead — let me know.
#
# Also note: the original notebook only saved Gemini's raw text evaluation to a .txt
# file and never parsed a numeric score out of it. Since the paper reports a numeric
# GScore, this version explicitly asks the model to end its answer with a line in the
# exact format "Score: X/10" and parses that out with a regex.
# *************************************************************************

import os
import re
import csv
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


GSCORE_PROMPT = (
    "Conduct a detailed evaluation of one modified image, labeled 'A' (Image 2), in "
    "comparison to an original image (Image 1). Image 1 serves as the baseline and "
    "will not be evaluated. Focus on assessing the quality of 'A' (Image 2), "
    "particularly in terms of its naturalness and the presence or absence of "
    "artifacts. Examine how well the edit preserves the integrity of the original "
    "image while introducing modifications. Look for any signs of distortions, "
    "unnatural colors, pixelation, or other visual inconsistencies. Rate the image "
    "on a scale from 1 to 10, where 10 represents excellent quality with seamless "
    "modifications, and 1 indicates poor quality with significant and noticeable "
    "artifacts. Provide a brief analysis, then end your response with a single line "
    "in EXACTLY this format (no extra text on that line): 'Score: X/10' where X is "
    "your numeric rating. Answer in English."
)

DEFAULT_MODEL_NAME = "gemini-2.0-flash"

_genai_module = None


def _load_gemini_client():
    """
    Lazily configure and return the `google.generativeai` module using
    GEMINI_API_KEY from the environment (loaded from a .env file if python-dotenv is
    installed and a .env file is present in the working directory).
    """
    global _genai_module
    if _genai_module is not None:
        return _genai_module

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; still works if GEMINI_API_KEY is already
              # set in the environment some other way.

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Create a .env file in the project root containing:\n"
            "    GEMINI_API_KEY=your-key-here\n"
            "Get a key at https://aistudio.google.com/apikey"
        )

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    _genai_module = genai
    return genai


def _extract_score(text: str):
    """Parse a 'Score: X/10' style line out of the model's response text."""
    match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # fallback: any "X/10" pattern anywhere in the text
    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b", text)
    if match:
        return float(match.group(1))
    return None


def compute_gscore(original_image_path, result_image_path,
                    model_name=DEFAULT_MODEL_NAME, retries=2, retry_delay=2.0):
    """
    Ask Gemini to rate `result_image_path` against `original_image_path` on a 1-10
    scale.

    Returns:
        (score, raw_text): score is a float in [1, 10] or None if it couldn't be
        parsed out of the response; raw_text is the model's full text response (or an
        error message string if every attempt failed).
    """
    genai = _load_gemini_client()
    model = genai.GenerativeModel(model_name)

    original_upload = genai.upload_file(str(original_image_path))
    result_upload = genai.upload_file(str(result_image_path))

    last_error = None
    last_text = ""
    for attempt in range(retries + 1):
        try:
            response = model.generate_content(
                [GSCORE_PROMPT, original_upload, result_upload],
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.4,
                    "top_p": 1,
                    "top_k": 32,
                },
            )
            last_text = response.text
            score = _extract_score(last_text)
            if score is not None:
                return score, last_text
        except Exception as e:
            last_error = e
        if attempt < retries:
            time.sleep(retry_delay)

    if last_error is not None:
        return None, f"GScore request failed: {last_error}"
    return None, last_text


def compute_average_gscore(dataset_path, result_path, output_csv="GScore.csv",
                            output_summary="GScore_summary.txt", model_name=DEFAULT_MODEL_NAME,
                            original_filename="original.jpg", result_filename="dragged_image.png"):
    """
    Batch-evaluate GScore over a dataset laid out like Drag100 / DragBench:
        dataset_path/<sample_name>/<original_filename>
        result_path/<sample_name>/<result_filename>
    Mirrors the folder-scanning convention used by compute_drag100_DAI.py.
    """
    dataset_dir = Path(dataset_path)
    result_dir = Path(result_path)
    scores = []

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SampleName', 'GScore', 'RawResponse'])

        for item in sorted(dataset_dir.iterdir()):
            if not item.is_dir():
                continue
            sample_name = item.name
            original_image_path = item / original_filename
            result_image_path = result_dir / sample_name / result_filename

            if not original_image_path.exists() or not result_image_path.exists():
                logging.warning(f"Skipping {sample_name}: missing original or result image.")
                continue

            score, raw_text = compute_gscore(original_image_path, result_image_path, model_name=model_name)
            if score is None:
                logging.warning(f"{sample_name}: could not parse a score from the response.")
            else:
                scores.append(score)
                logging.info(f"{sample_name}: GScore = {score:.2f}")

            writer.writerow([sample_name, score if score is not None else '', raw_text.replace('\n', ' ')])

    if scores:
        average = sum(scores) / len(scores)
        with open(output_summary, 'w') as f:
            f.write(f"Average GScore: {average:.4f}\n")
            f.write(f"Evaluated samples: {len(scores)}\n")
        logging.info(f"Average GScore: {average:.4f} over {len(scores)} samples.")
    else:
        logging.warning("No scores were computed; nothing to average.")


def main():
    parser = argparse.ArgumentParser(description="Batch GScore evaluation (LLM-as-judge quality scoring).")
    parser.add_argument('--dataset_path', required=True, help="Root folder containing per-sample subfolders with the original image.")
    parser.add_argument('--result_path', required=True, help="Root folder containing per-sample subfolders with the dragged/result image.")
    parser.add_argument('--original_filename', default='original.jpg')
    parser.add_argument('--result_filename', default='dragged_image.png')
    parser.add_argument('--output_csv', default='GScore.csv')
    parser.add_argument('--output_summary', default='GScore_summary.txt')
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    compute_average_gscore(
        args.dataset_path, args.result_path,
        output_csv=args.output_csv, output_summary=args.output_summary,
        model_name=args.model_name,
        original_filename=args.original_filename, result_filename=args.result_filename,
    )


if __name__ == '__main__':
    main()