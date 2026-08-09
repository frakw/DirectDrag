# *************************************************************************
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

import os
import gradio as gr
from utils.ui_utils import (
    get_points, undo_point, undo_pair, show_cur_points,
    clear_all, store_img, train_lora_interface, save_image_mask_points, save_drag_result,
    save_intermediate_images, create_video, resize_image_max_dim,
    compute_image_hash, get_lora_cache_dir, lora_cache_is_trained, touch_lora_cache
)

from directdrag import run_directdrag

import numpy as np

import cv2

import hashlib

import json

from PIL import Image

from utils.eval_utils import compute_if, compute_md
from gscore import compute_gscore

LENGTH = 512
MAX_LORA_CACHE_ENTRIES = 10
DEMO_SAMPLES_DIR = "demo_samples"


LOGO_PATH = "logo.png"
GITHUB_URL = "https://github.com/frakw/DirectDrag"
PROJECT_PAGE_URL = "https://frakw.github.io/DirectDrag/"
PAPER_URL = "https://arxiv.org/pdf/2512.03981"


def _logo_data_uri():
    """Read logo.png and encode as a base64 data URI so it can be embedded directly
    in an <img> tag without relying on Gradio's static file serving (which can crop
    or fail to load local paths depending on version/config)."""
    import base64
    if not os.path.exists(LOGO_PATH):
        return None
    with open(LOGO_PATH, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def create_markdown_section():
    logo_uri = _logo_data_uri()
    logo_html = (
        f'<img src="{logo_uri}" alt="DirectDrag logo" '
        f'style="height:40px;width:auto;vertical-align:middle;margin-right:10px;">'
        if logo_uri else ""
    )

    # NOTE: previously this used a real <h1> wrapping everything, but Gradio's theme
    # CSS applies typography rules (e.g. inside the "prose" class) to heading tags
    # that can override inline styles like white-space:nowrap with higher specificity,
    # which is why the GitHub/Project Page links kept wrapping onto a second line even
    # with white-space:nowrap set. Using plain <div>/<span> tags (styled to look like a
    # heading) avoids inheriting those heading-specific rules, and "!important" is
    # added as a second safety net against any remaining theme overrides.
    gr.HTML(f"""
<div style="display:flex;align-items:center;flex-wrap:nowrap !important;gap:10px;margin-bottom:4px;">
  {logo_html}<span style="font-size:32px;font-weight:700;white-space:nowrap !important;">DirectDrag ✨</span>
  <span style="display:inline-flex;align-items:center;gap:6px;font-size:16px;font-weight:normal;margin-left:12px;white-space:nowrap !important;flex-shrink:0;">
    <a href="{GITHUB_URL}" target="_blank" style="text-decoration:none;">
      <img src="https://img.shields.io/badge/GitHub-DirectDrag-181717?logo=github" alt="GitHub" style="vertical-align:middle;">
    </a>
    &nbsp;|&nbsp;
    <a href="{PROJECT_PAGE_URL}" target="_blank">🌐 Project Page</a>
    &nbsp;|&nbsp;
    <a href="{PAPER_URL}" target="_blank">📄 Paper</a>
  </span>
</div>
""")

    gr.Markdown("""
**DirectDrag: High-Fidelity, Mask-Free, Prompt-Free Drag-based Image Editing via Readout-Guided Feature Alignment**

👋 Welcome to DirectDrag! Follow these steps to easily manipulate your images:

1. **Upload an image**: Upload your image on the left under "Original Image & Click Points".
2. **Mark drag points**: Click directly on the image. Points alternate as pairs — "handle point" (red) then "target point" (blue). You can add multiple point pairs.
3. **(Optional) Tune parameters**: Adjust the DirectDrag / Drag / LoRA / Advanced parameters in the tabs below. The defaults already work well for most images.
4. **Run**: Click "Run" to start editing. The first time you run on a given image, a dedicated LoRA is trained automatically (it won't be retrained the next time you run on the same image). Oversized images are automatically downscaled (longest side capped at 512px) before running, to avoid running out of memory.
5. **Manage points**: Made a mistake? Use "Undo Point" to remove the last point, or "Undo Pair" to remove the last handle/target pair.

    """)


def create_base_model_config_ui():
    with gr.Tab("Diffusion Model"):
        with gr.Row():
            local_models_dir = 'local_pretrained_models'
            os.makedirs(local_models_dir, exist_ok=True)
            local_models_choice = \
                [os.path.join(local_models_dir, d) for d in os.listdir(local_models_dir) if
                 os.path.isdir(os.path.join(local_models_dir, d))]
            model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                                     label="Diffusion Model Path",
                                     choices=[
                                                 "runwayml/stable-diffusion-v1-5",
                                                 "stabilityai/stable-diffusion-2-1-base",
                                                 "stabilityai/stable-diffusion-xl-base-1.0",
                                             ] + local_models_choice
                                     )
            vae_path = gr.Dropdown(value="stabilityai/sd-vae-ft-mse",
                                   label="VAE choice",
                                   choices=["stabilityai/sd-vae-ft-mse",
                                            "default"] + local_models_choice
                                   )

    return model_path, vae_path


def create_lora_parameters_ui():
    with gr.Tab("LoRA Parameters"):
        with gr.Row():
            lora_path = gr.Textbox(value=f"./lora_tmp", label="LoRA Cache Directory",
                                   info=f"Base folder for per-image LoRA cache. Each uploaded image gets its own "
                                        f"subfolder (keyed by content hash) so the same image is never retrained; "
                                        f"only the {MAX_LORA_CACHE_ENTRIES} most recently used images are kept, "
                                        f"older ones are deleted automatically.",
                                   placeholder="Enter path for LoRA cache")
            lora_step = gr.Number(value=70, label="LoRA training steps", precision=0)
            lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
            lora_batch_size = gr.Number(value=4, label="LoRA batch size", precision=0)
            lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)
    return lora_path, lora_step, lora_lr, lora_batch_size, lora_rank


def _ensure_placeholder_result_file():
    """
    Create (if missing) and return the path to a tiny placeholder image at
    gradio_results/result.png. Used as the initial `value` for download_result_file
    so that component is never in Gradio's large, broken-looking "empty" drop-zone
    state — it always renders as a compact file card instead. Before a real result
    exists, clicking download just re-downloads this placeholder, which is an
    accepted no-op trade-off (requested fix, since CSS alone couldn't reliably shrink
    the empty state).
    """
    save_dir = "gradio_results"
    os.makedirs(save_dir, exist_ok=True)
    placeholder_path = os.path.join(save_dir, "result.png")
    if not os.path.exists(placeholder_path):
        Image.new("RGB", (1, 1), color=(255, 255, 255)).save(placeholder_path)
    return placeholder_path


def create_real_image_editing_ui():
  
    with gr.Row():

        '''
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>📤 Draw Mask</h2>")
            canvas = gr.Image(type="numpy", tool="sketch", label="Draw your mask on the image",
                              show_label=True, height=LENGTH, width=LENGTH)  # for mask painting
            with gr.Row():
                train_lora_button = gr.Button("Train LoRA")
                lora_path = gr.Textbox(value=f"./lora_data/test", label="LoRA Path",
                                       placeholder="Enter path for LoRA data")

            with gr.Row():
                lora_status_bar = gr.Textbox(label="LoRA Training Status", interactive=False)
        '''

        with gr.Column():
            gr.Markdown("<h2 style='text-align: left;'>Original Image & Click Points</h2>")
            # Reverted to the exact sizing from the original, known-good baseline
            # (fixed square height=width=LENGTH) after two attempts at a non-square,
            # half-width layout both resulted in a broken/unfit display that I
            # couldn't debug further without a live browser to inspect the rendered
            # CSS. If a non-square layout is still wanted, it'll need live devtools
            # inspection to get right.
            input_image = gr.Image(type="numpy", label="Click on the image to mark points",
                                show_label=True, height=LENGTH, width=LENGTH)  # for points clicking
            with gr.Row():
                undo_point_button = gr.Button("Undo Point")
                undo_pair_button = gr.Button("Undo Pair")
                
                #save_button = gr.Button('Save Current Data')
                #data_dir = gr.Textbox(value='./dataset/test', label="Data Directory",
                #                      placeholder="Enter directory path for mask and points")
            

        with gr.Column():
            gr.Markdown("<h2 style='text-align: left;'>Dragged Image</h2>")
            # elem_id so we can target this exact box with custom CSS (see the
            # gr.Blocks(css=...) in main()) to fix the output image sticking to the
            # top instead of being vertically centered like the input image.
            output_image = gr.Image(type="numpy", label="View the editing results here",
                                    show_label=True, height=LENGTH, width=LENGTH,
                                    elem_id="center-img")
            # equal_height=False so this Row does NOT stretch the Run button to match
            # gr.File's much taller default drop-zone height (Gradio Rows default to
            # stretching all children to the tallest one's height) — this is what made
            # Run balloon up to match the Download Result File box in the screenshot.
            with gr.Row(equal_height=False):
                run_button = gr.Button("Run", scale=1, interactive=False)
                # A dedicated download component for the result image, separate from
                # Gradio's built-in download icon on the Image component itself. The
                # built-in icon always seems to use a generic filename like "image.png"
                # regardless of what we return from Python (couldn't get it to respect
                # a custom name), so this File card is the reliable way to download the
                # result with its correct, hash-based filename.
                #
                # Pre-seeded with a placeholder file (see _ensure_placeholder_result_file
                # below) so the component is never in Gradio's "empty" state, which has
                # an oversized drop-zone we couldn't reliably shrink with CSS. Before a
                # real result exists, this button just re-downloads the placeholder
                # (effectively a no-op) — that's an accepted trade-off.
                download_result_file = gr.File(label="Download Result Image (correct filename)",
                                               interactive=False, scale=1,
                                               elem_id="download-result-file",
                                               value=_ensure_placeholder_result_file())
                #clear_all_button = gr.Button("Clear All")
                #save_result = gr.Button("Save Result")
                #show_points = gr.Button("Show Points")
                #result_save_path = gr.Textbox(value='./result/test', label="Result Folder",
                #                            placeholder="Enter path to save the results")

    # Full-width row containing two halves: left = Drag Instruction (view / edit /
    # import / download / load-from-file), right = reserved space for a future
    # evaluation results panel (IF / MD / GScore).
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='margin-bottom:4px;'>Drag Instruction</h3>")
            with gr.Row():
                instruction_file_picker = gr.File(label="Load Drag Instruction (.json)",
                                                  file_types=[".json"], scale=1)
                drag_instruction_textbox = gr.Textbox(
                    label="Drag Instruction (JSON)",
                    value='{\n  "points": []\n}',
                    lines=4, max_lines=8,
                    info='Format: {"points": [[x1,y1],[x2,y2],...]} — alternating handle/target pairs. '
                         'Edit freely, then click "Import" to apply.',
                    scale=1
                )
            with gr.Row():
                import_instruction_button = gr.Button("Import / Apply")
                download_instruction_button = gr.Button("Download Instruction")
            drag_instruction_status = gr.Markdown("")
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='margin-bottom:4px;'>Evaluation</h3>")
            with gr.Row():
                eval_if_checkbox = gr.Checkbox(label="IF", value=False)
                eval_md_checkbox = gr.Checkbox(label="MD", value=False)
                eval_gscore_checkbox = gr.Checkbox(label="GScore", value=False)
            eval_result_textbox = gr.Textbox(
                label="Evaluation Result",
                lines=4, max_lines=8,
                interactive=False,
                info="IF/MD load extra models the first time they're used and can be "
                     "slow; GScore calls an external API (needs GEMINI_API_KEY)."
            )
            # Disabled until a result image exists (enabled by attach_run_button_event
            # once run_direct_drag_step finishes).
            evaluate_button = gr.Button("Evaluate", interactive=False)

    # Full-width row (not nested inside a Column) so the LoRA training progress bar,
    # which Gradio overlays on top of this component while `progress.tqdm()` is
    # running, spans the entire horizontal width of the app instead of just one
    # half of it. It stays empty/invisible when idle since we removed the old
    # separate "LoRA Training Status" textbox.
    with gr.Row():
        lora_progress_display = gr.HTML(value="")

    return input_image, undo_point_button, undo_pair_button, \
           output_image, run_button, lora_progress_display, \
           drag_instruction_textbox, import_instruction_button, download_instruction_button, \
           drag_instruction_status, instruction_file_picker, download_result_file, \
           eval_if_checkbox, eval_md_checkbox, eval_gscore_checkbox, \
           eval_result_textbox, evaluate_button


def create_drag_parameters_ui():
    with gr.Tab("Drag Parameters"):
        with gr.Row():
            latent_lr = gr.Number(value=0.02, label="Learning rate")
            drag_end_step = gr.Number(value=7, label="End time step", precision=0)
            drag_per_step = gr.Number(value=10, label="Point tracking number per each step", precision=0)

    return latent_lr, drag_end_step, drag_per_step


def create_directdrag_parameters_ui():
    with gr.Tab("DirectDrag Parameters"):
        with gr.Row():
            enable_soft_mask = gr.Checkbox(label="Enable Soft Mask", value=True, interactive=True)
            enable_readout_guided_feature_alignment = gr.Checkbox(label="Enable Readout-Guided Feature Alignment", value=True, interactive=True)
            enable_latent_warpage_function = gr.Checkbox(label="Enable Latent Warpage Function", value=True, interactive=True)
        with gr.Row():
            soft_mask_sigma = gr.Number(value=30, label="Soft mask sigma", minimum=10, maximum=100, step=1, precision=0, interactive=True)
            readout_guided_feature_alignment_multiplier = gr.Number(value=350, label="Readout-Guided Feature Alignment Multiplier", minimum=100, maximum=500, step=50, precision=0, interactive=True)
            latent_warpage_function_ratio = gr.Number(value=0.15, label="Latent Warpage Function Ratio", minimum=0.01, maximum=0.5, step=0.01, precision=2, interactive=True)

    return enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function, \
           soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio


def create_advance_parameters_ui():
    with gr.Tab("Advanced Parameters"):
        with gr.Row():
            r1 = gr.Number(value=4, label="Motion supervision feature path size", precision=0)
            r2 = gr.Number(value=12, label="Point tracking feature patch size", precision=0)
            drag_distance = gr.Number(value=4, label="The distance for motion supervision", precision=0)
            feature_idx = gr.Number(value=3, label="The index of the features [0,3]", precision=0)
            max_drag_per_track = gr.Number(value=3,
                                           label="Motion supervision times for each point tracking",
                                           precision=0)

        with gr.Row():
            lam = gr.Number(value=0.2, label="Lambda", info="Regularization strength on unmasked areas")
            inversion_strength = gr.Slider(0, 1.0,
                                           value=0.75,
                                           label="Inversion strength")
            max_track_no_change = gr.Number(value=10, label="Early stop",
                                            info="The maximum number of times points is unchanged.")

    return (r1, r2, drag_distance, feature_idx, max_drag_per_track, lam,
            inversion_strength, max_track_no_change)


def create_intermediate_save_ui():
    with gr.Tab("Get Intermediate Images"):
        with gr.Row():
            save_intermediates_images = gr.Checkbox(label='Save intermediate images')
            get_mp4 = gr.Button("Get video")

    return save_intermediates_images, get_mp4

    
def m_store_img(img, length=512):
    print('m_store_img')
    print(type(img))
    image = img["image"]
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    image = np.array(image)
    return image

def attach_canvas_event(canvas: gr.State, original_image: gr.State,
                        selected_points: gr.State, input_image):
    canvas.edit(
        m_store_img,
        [canvas],
        [original_image, selected_points, input_image]
    )


'''
def attach_input_image_event(input_image, selected_points):
    input_image.select(
        get_points,
        [input_image, selected_points],
        [input_image]
    )
'''

def points_to_instruction_json(points):
    """Serialize the current handle/target points into the Drag Instruction JSON format."""
    normalized = [[int(p[0]), int(p[1])] for p in points]
    return json.dumps({"points": normalized}, indent=2)


def compute_run_button_state(image, points):
    """Run should only be clickable once there's an image AND at least one complete
    handle/target point pair."""
    can_run = image is not None and len(points) >= 2
    return gr.update(interactive=can_run)


def sync_instruction_display(points, original_image):
    """Refresh the Drag Instruction textbox, clear any stale status message, and
    enable/disable the Run button — all whenever the points change (upload / click /
    undo)."""
    return points_to_instruction_json(points), "", compute_run_button_state(original_image, points)


def attach_input_image_event(input_image, selected_points, original_image, drag_instruction_textbox=None, drag_instruction_status=None, run_button=None):
    def upload_store_img(input_image):
        # Downscale here (at upload time) so every later click/select event only ever
        # has to re-encode and transfer a small image back to the browser. Without
        # this, uploading a high-resolution photo makes every single click extremely
        # slow, because get_points() below returns the *entire* currently-displayed
        # image on every click, and Gradio has to re-serialize/send that whole image
        # back over the network each time.
        from utils.ui_utils import resize_image_max_dim
        resized = resize_image_max_dim(input_image, max_dim=LENGTH)
        return resized.copy(), []

    def refresh_input_image_display(resized_image):
        # Deliberately a SEPARATE event (via .then(), run strictly after the upload
        # event above has fully committed) rather than an extra output on the upload
        # event itself. Making input_image both the trigger and an output of the very
        # same event caused a race condition in Gradio 3.50.2 where the widget got
        # stuck showing a broken/error state until a different event (like a click)
        # forced a re-render. Chaining a distinct .then() event avoids that.
        return resized_image

    upload_event = input_image.upload(
        fn=upload_store_img,
        inputs=[input_image],
        outputs=[original_image, selected_points]
    ).then(
        fn=refresh_input_image_display,
        inputs=[original_image],
        outputs=[input_image]
    )

    select_event = input_image.select(
        fn=get_points,
        inputs=[input_image, selected_points],
        outputs=[input_image]
    )

    # Additive only: keep the Drag Instruction textbox + Run button state in sync with
    # the points, without altering anything about the upload/select events above.
    if drag_instruction_textbox is not None and drag_instruction_status is not None and run_button is not None:
        upload_event.then(
            fn=sync_instruction_display,
            inputs=[selected_points, original_image],
            outputs=[drag_instruction_textbox, drag_instruction_status, run_button]
        )
        select_event.then(
            fn=sync_instruction_display,
            inputs=[selected_points, original_image],
            outputs=[drag_instruction_textbox, drag_instruction_status, run_button]
        )

def attach_undo_button_event(undo_point_button, undo_pair_button, original_image, selected_points, input_image, drag_instruction_textbox=None, drag_instruction_status=None, run_button=None):
    undo_point_event = undo_point_button.click(
        undo_point,
        [original_image, selected_points],
        [input_image, selected_points]
    )
    undo_pair_event = undo_pair_button.click(
        undo_pair,
        [original_image, selected_points],
        [input_image, selected_points]
    )

    # Additive only: keep the Drag Instruction textbox + Run button state in sync
    # after undo, without altering the undo_point/undo_pair calls above.
    if drag_instruction_textbox is not None and drag_instruction_status is not None and run_button is not None:
        undo_point_event.then(
            fn=sync_instruction_display,
            inputs=[selected_points, original_image],
            outputs=[drag_instruction_textbox, drag_instruction_status, run_button]
        )
        undo_pair_event.then(
            fn=sync_instruction_display,
            inputs=[selected_points, original_image],
            outputs=[drag_instruction_textbox, drag_instruction_status, run_button]
        )

'''
def attach_train_lora_button_event(train_lora_button, original_image, prompt,
                                   model_path, vae_path, lora_path,
                                   lora_step, lora_lr, lora_batch_size, lora_rank,
                                   lora_status_bar):
    train_lora_button.click(
        train_lora_interface,
        [original_image, prompt, model_path, vae_path, lora_path,
         lora_step, lora_lr, lora_batch_size, lora_rank],
        [lora_status_bar]
    )
'''

def run_lora_training_step(original_image, model_path, vae_path,
         lora_path, lora_step, lora_lr, lora_batch_size, lora_rank,
         progress=gr.Progress()):
    # NOTE: this function is bound directly to run_button.click() as the FIRST of two
    # chained events (see attach_run_button_event). Gradio only wires up a live
    # progress bar for a parameter with a gr.Progress() default on the function that is
    # *directly* bound to the event, so `progress` is accepted here and explicitly
    # forwarded into train_lora_interface -> train_lora below, instead of letting
    # train_lora_interface create its own unbound gr.Progress() (which never shows up
    # in the UI).
    #
    # This step is deliberately split out from the actual drag-editing step
    # (run_direct_drag_step) and only outputs to `lora_progress_display`, a dedicated
    # full-width HTML component. That way:
    #   - the LoRA training progress bar overlay spans the full width of the app
    #     (Gradio overlays progress on top of an event's output components), instead
    #     of only covering half the screen.
    #   - the "Dragged Image" output column is not part of this event's outputs, so it
    #     never shows the LoRA training progress overlay.
    #
    # LoRA caching: `lora_path` (from the "LoRA Cache Directory" field) is treated as a
    # BASE directory. Each distinct uploaded image gets its own subfolder named after a
    # content hash of the image, so re-running on the exact same image reuses the
    # already-trained LoRA instead of retraining. Only the MAX_LORA_CACHE_ENTRIES most
    # recently used images are kept; older cache folders are deleted automatically.
    if original_image is None:
        return "", lora_path

    image_hash = compute_image_hash(original_image)
    cache_dir = get_lora_cache_dir(lora_path, image_hash)

    if not lora_cache_is_trained(cache_dir):
        train_lora_interface(original_image, "", model_path, vae_path,
                              cache_dir, lora_step, lora_lr, lora_batch_size, lora_rank,
                              progress=progress)

    touch_lora_cache(lora_path, image_hash, max_entries=MAX_LORA_CACHE_ENTRIES)
    return "", cache_dir


def run_direct_drag_step(original_image, input_image, selected_points,
         inversion_strength, lam, latent_lr, model_path, vae_path,
         lora_path, drag_end_step, drag_per_step, r1, r2, d,
         max_drag_per_track, max_track_no_change, feature_idx, save_intermediates_images,
         enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function,
         soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio):
    # NOTE: `lora_path` here is the *resolved* per-image cache directory produced by
    # run_lora_training_step above (passed in via the resolved_lora_path_state State),
    # not the raw "LoRA Cache Directory" base path from the UI field.
    result_save_path = "gradio_results"
    os.makedirs(result_save_path, exist_ok=True)  # ensure the save directory exists
    out_image, new_points = run_directdrag(original_image, input_image, selected_points,
                        inversion_strength, lam, latent_lr, model_path, vae_path,
                        lora_path, drag_end_step, drag_per_step, r1, r2, d,
                        max_drag_per_track, max_track_no_change, feature_idx,
                        result_save_path, save_intermediates_images,
                        save_intermedia=False, compare_mode=False, once_drag=False,
                        enable_soft_mask=enable_soft_mask, enable_readout_guided_feature_alignment=enable_readout_guided_feature_alignment, enable_latent_warpage_function=enable_latent_warpage_function,
                        soft_mask_sigma=soft_mask_sigma, readout_guided_feature_alignment_multiplier=readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio=latent_warpage_function_ratio)

    # Name the saved result after the input image, e.g. "a1b2c3d4_dragged.png".
    # NOTE: Gradio's numpy-typed Image component does not preserve the browser's
    # original uploaded filename, so we derive a stable name from a content hash of
    # the input image instead. Running again on the exact same image overwrites this
    # file (same hash -> same filename).
    image_hash = hashlib.md5(np.ascontiguousarray(original_image).tobytes()).hexdigest()[:8]
    result_save_filename = os.path.join(result_save_path, f"{image_hash}_dragged.png")
    out_image_bgr = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    print(result_save_filename)
    cv2.imwrite(result_save_filename, out_image_bgr)
    # `out_image` (numpy array) is returned for the on-screen preview; the same file
    # is also returned as a filepath for the separate `download_result_file` File
    # component, which is what actually gives the downloaded file the correct name
    # (Gradio's built-in Image download icon doesn't reliably respect it).
    return out_image, new_points, result_save_filename

def attach_run_button_event(run_button, original_image, input_image,
                            selected_points, inversion_strength, lam, latent_lr,
                            model_path, vae_path, 
                            lora_path, lora_step, lora_lr, lora_batch_size, lora_rank,
                            drag_end_step, drag_per_step,
                            output_image, r1, r2, d, feature_idx, new_points,
                            max_drag_per_track, max_track_no_change,
                            save_intermediates_images,
                            enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function,
                            soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio,
                            lora_progress_display, resolved_lora_path_state, download_result_file,
                            evaluate_button):
    run_button.click(
        run_lora_training_step,
        [original_image, model_path, vae_path,
         lora_path, lora_step, lora_lr, lora_batch_size, lora_rank],
        [lora_progress_display, resolved_lora_path_state]
    ).then(
        run_direct_drag_step,
        [original_image, input_image, selected_points,
         inversion_strength, lam, latent_lr, model_path, vae_path,
         resolved_lora_path_state, drag_end_step, drag_per_step, r1, r2, d,
         max_drag_per_track, max_track_no_change, feature_idx, save_intermediates_images,
         enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function,
         soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio],
        [output_image, new_points, download_result_file]
    ).then(
        # Only enable the Evaluate button once a result image actually exists.
        fn=lambda: gr.update(interactive=True),
        inputs=[],
        outputs=[evaluate_button]
    )


# -------------------- Evaluation: IF / MD / GScore --------------------

def run_evaluation(eval_if, eval_md, eval_gscore, original_image, current_output_image, selected_points):
    if original_image is None or current_output_image is None:
        return "No result image to evaluate yet — run DirectDrag first."
    if not (eval_if or eval_md or eval_gscore):
        return "Select at least one of IF / MD / GScore to evaluate."

    lines = []

    if eval_if:
        try:
            score = compute_if(original_image, current_output_image)
            lines.append(f"IF (1 - LPIPS): {score:.4f}")
        except Exception as e:
            lines.append(f"IF: error — {e}")

    if eval_md:
        try:
            md = compute_md(original_image, current_output_image, selected_points, prompt="")
            lines.append(f"MD: {md:.2f}" if md is not None else "MD: N/A (no complete point pairs)")
        except Exception as e:
            lines.append(f"MD: error — {e}")

    if eval_gscore:
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                original_path = os.path.join(tmp_dir, "original.png")
                result_path = os.path.join(tmp_dir, "result.png")
                Image.fromarray(original_image).save(original_path)
                Image.fromarray(current_output_image).save(result_path)
                score, _raw_text = compute_gscore(original_path, result_path)
            lines.append(f"GScore: {score:.1f}/10" if score is not None else "GScore: could not parse a score from the response")
        except Exception as e:
            lines.append(f"GScore: error — {e}")

    return "\n".join(lines)


def attach_evaluate_button_event(evaluate_button, eval_if_checkbox, eval_md_checkbox, eval_gscore_checkbox,
                                 original_image, output_image, selected_points, eval_result_textbox):
    evaluate_button.click(
        run_evaluation,
        [eval_if_checkbox, eval_md_checkbox, eval_gscore_checkbox, original_image, output_image, selected_points],
        [eval_result_textbox]
    )




def attach_show_points_event(show_points, output_image, selected_points):
    show_points.click(
        show_cur_points,
        [output_image, selected_points],
        [output_image]
    )

def attach_clear_all_button_event(clear_all_button, input_image,
                                  output_image, selected_points, original_image):
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [input_image, output_image, selected_points, original_image]
    )


# -------------------- Drag Instruction: view / download / import --------------------

def apply_drag_instruction(instruction_text, original_image, selected_points):
    """
    Parse the JSON in the Drag Instruction textbox and, if valid, replace the current
    points with it (redrawing them on the image). Only the points are stored/loaded —
    no image data — so we validate the parsed points against the *currently loaded*
    image's dimensions to catch instructions that don't belong to this image.
    """
    try:
        data = json.loads(instruction_text)
        raw_points = data["points"]
        points = [[int(p[0]), int(p[1])] for p in raw_points]
        assert all(len(p) == 2 for p in points)
    except Exception:
        return gr.update(), selected_points, gr.update(), "⚠️ Drag instruction format not match", \
               compute_run_button_state(original_image, selected_points)

    if original_image is None:
        return gr.update(), selected_points, gr.update(), "⚠️ Please upload an image first", \
               compute_run_button_state(original_image, selected_points)

    height, width = original_image.shape[:2]
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            return gr.update(), selected_points, gr.update(), "⚠️ Instruction not suited for this image", \
                   compute_run_button_state(original_image, selected_points)

    new_display = show_cur_points(original_image.copy(), [tuple(p) for p in points])
    normalized_text = points_to_instruction_json(points)
    return new_display, points, normalized_text, "✅ Instruction applied", \
           compute_run_button_state(original_image, points)


def load_instruction_file(file_obj):
    """
    Reads the content of an uploaded .json file (from the "Load Drag Instruction"
    picker) into the Drag Instruction textbox. Only loads the raw text here — the
    actual points get applied via the same apply_drag_instruction() path used by the
    "Import / Apply" button (chained with .then() by attach_drag_instruction_events),
    so validation/error messages stay consistent regardless of whether the JSON was
    typed, pasted, or picked from a file.
    """
    if file_obj is None:
        return gr.update()
    try:
        with open(file_obj.name, 'r') as f:
            return f.read()
    except Exception:
        return gr.update()


def attach_drag_instruction_events(import_instruction_button, download_instruction_button,
                                    drag_instruction_textbox, drag_instruction_status,
                                    original_image, selected_points, input_image,
                                    instruction_file_picker, run_button):
    import_instruction_button.click(
        apply_drag_instruction,
        [drag_instruction_textbox, original_image, selected_points],
        [input_image, selected_points, drag_instruction_textbox, drag_instruction_status, run_button]
    )

    # Picking a .json file loads its text into the textbox, then immediately runs the
    # same apply step as clicking "Import / Apply" — so choosing a file is a one-step
    # "load and apply" action.
    instruction_file_picker.upload(
        load_instruction_file,
        [instruction_file_picker],
        [drag_instruction_textbox]
    ).then(
        apply_drag_instruction,
        [drag_instruction_textbox, original_image, selected_points],
        [input_image, selected_points, drag_instruction_textbox, drag_instruction_status, run_button]
    )

    # Pure client-side download: builds a .json file from the textbox's current
    # content and triggers a normal browser download, with no extra visible
    # component / intermediate "file card" step and no round trip to the server.
    download_instruction_button.click(
        fn=None,
        inputs=[drag_instruction_textbox],
        outputs=[],
        _js="""
        (instructionText) => {
            const blob = new Blob([instructionText], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'drag_instruction.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            return [];
        }
        """
    )


# -------------------- Example images (click to load image + saved points) --------------------

def load_example(_current_input_image, image_path):
    """
    Loads a demo_samples/ example image and, if a matching JSON file with the same
    base filename exists alongside it (e.g. "candy.png" -> "candy.json", produced via
    the Download Instruction button), also loads and applies its saved points.
    Silently falls back to "no points" if there's no matching JSON, and reports the
    same validation messages as apply_drag_instruction() if the JSON doesn't match
    this image's dimensions or is malformed.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    img = resize_image_max_dim(img, max_dim=LENGTH)

    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(os.path.dirname(image_path), base + ".json")

    points = []
    status = ""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            raw_points = [[int(p[0]), int(p[1])] for p in data["points"]]
            height, width = img.shape[:2]
            if all(0 <= x < width and 0 <= y < height for x, y in raw_points):
                points = raw_points
            else:
                status = "⚠️ Instruction not suited for this image"
        except Exception:
            status = "⚠️ Drag instruction format not match"

    display_img = show_cur_points(img.copy(), [tuple(p) for p in points]) if points else img.copy()
    instruction_text = points_to_instruction_json(points)
    run_state = compute_run_button_state(img, points)
    return img, points, display_img, instruction_text, status, run_state


def create_examples_ui(input_image, original_image, selected_points,
                       drag_instruction_textbox, drag_instruction_status, run_button):
    if not os.path.isdir(DEMO_SAMPLES_DIR):
        return

    example_paths = sorted(
        os.path.join(DEMO_SAMPLES_DIR, fname)
        for fname in os.listdir(DEMO_SAMPLES_DIR)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not example_paths:
        return

    # A hidden Textbox carries the raw file path string into load_example() (so it can
    # look up a matching .json file by name), while `input_image` (an Image component)
    # is what makes gr.Examples render nice image thumbnails instead of a text table.
    example_path_holder = gr.Textbox(visible=False)
    gr.Examples(
        examples=[[p, p] for p in example_paths],
        inputs=[input_image, example_path_holder],
        fn=load_example,
        outputs=[original_image, selected_points, input_image, drag_instruction_textbox, drag_instruction_status, run_button],
        run_on_click=True,
        label="Examples (click a thumbnail to load it; its saved points load automatically if a matching .json file exists next to it in demo_samples/)",
    )

'''
def attach_save_button_event(save_button, mask, selected_points, input_image, save_dir):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_button.click(
        save_image_mask_points,
        inputs=[mask, selected_points, input_image, save_dir],
        outputs=[]
    )
'''

'''
def attach_save_result_event(save_result, output_image, new_points, result_path):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_result.click(
        save_drag_result,
        inputs=[output_image, new_points, result_path],
        outputs=[]
    )
'''

'''
def attach_video_event(get_mp4_button, result_save_path, data_dir):
    get_mp4_button.click(
        create_video,
        inputs=[result_save_path, data_dir]
    )
'''

def main():
    # Custom CSS: fix the output result image sticking to the top of its box instead
    # of being vertically centered like the input image. Targets elem_id="center-img"
    # set on output_image (see create_real_image_editing_ui). Still unverified against
    # a live render — this is the approach requested; if it doesn't take effect we
    # need the actual live DOM (via browser "Inspect Element", not "View Page Source")
    # to write a selector that matches Gradio's real generated structure.
    custom_css = """
    #center-img {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #center-img img {
        margin: 0 auto;
    }
    """
    with gr.Blocks(css=custom_css) as demo:
        selected_points = gr.State([])
        new_points = gr.State([])
        original_image = gr.State(value=None)
        resolved_lora_path_state = gr.State(value=None)
        create_markdown_section()
        intermediate_images = gr.State([])

        input_image, undo_point_button, undo_pair_button, \
        output_image, run_button, lora_progress_display, \
        drag_instruction_textbox, import_instruction_button, download_instruction_button, \
        drag_instruction_status, instruction_file_picker, download_result_file, \
        eval_if_checkbox, eval_md_checkbox, eval_gscore_checkbox, \
        eval_result_textbox, evaluate_button = create_real_image_editing_ui()

        enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function, \
           soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio = create_directdrag_parameters_ui()
        
        latent_lr, drag_end_step, drag_per_step = create_drag_parameters_ui()

        model_path, vae_path = create_base_model_config_ui()
        lora_path, lora_step, lora_lr, lora_batch_size, lora_rank = create_lora_parameters_ui()
        r1, r2, d, feature_idx, max_drag_per_track, lam, inversion_strength, max_track_no_change = \
            create_advance_parameters_ui()
        save_intermediates_images, get_mp4_button = create_intermediate_save_ui()

        attach_input_image_event(input_image, selected_points, original_image,
                                 drag_instruction_textbox, drag_instruction_status, run_button)
        attach_undo_button_event(undo_point_button, undo_pair_button, original_image, selected_points, input_image,
                                 drag_instruction_textbox, drag_instruction_status, run_button)
        attach_drag_instruction_events(import_instruction_button, download_instruction_button,
                                       drag_instruction_textbox, drag_instruction_status,
                                       original_image, selected_points, input_image,
                                       instruction_file_picker, run_button)
        attach_run_button_event(run_button, original_image, input_image, selected_points,
                                inversion_strength, lam, latent_lr, model_path, vae_path, 
                                lora_path, lora_step, lora_lr, lora_batch_size, lora_rank,
                                drag_end_step, drag_per_step, output_image,
                                r1, r2, d, feature_idx, new_points, max_drag_per_track,
                                max_track_no_change, save_intermediates_images,
                                enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function,
                                soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio,
                                lora_progress_display, resolved_lora_path_state, download_result_file,
                                evaluate_button)
        attach_evaluate_button_event(evaluate_button, eval_if_checkbox, eval_md_checkbox, eval_gscore_checkbox,
                                     original_image, output_image, selected_points, eval_result_textbox)
        #attach_show_points_event(show_points, output_image, new_points)
        #attach_clear_all_button_event(clear_all_button, input_image, output_image, selected_points,
        #                              original_image)

        # Placed last so it renders at the very bottom of the page.
        create_examples_ui(input_image, original_image, selected_points,
                           drag_instruction_textbox, drag_instruction_status, run_button)

    demo.queue().launch(share=False, debug=True)


if __name__ == '__main__':
    main()