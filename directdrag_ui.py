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
    save_intermediate_images, create_video
)

from directdrag import run_directdrag

import numpy as np

import cv2

import hashlib

LENGTH = 512
IMAGE_BOX_HEIGHT = 480  # shared fixed height for input_image / output_image boxes


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
            lora_path = gr.Textbox(value=f"./lora_tmp", label="LoRA Path", placeholder="Enter path for LoRA data")
            lora_step = gr.Number(value=70, label="LoRA training steps", precision=0)
            lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
            lora_batch_size = gr.Number(value=4, label="LoRA batch size", precision=0)
            lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)
    return lora_path, lora_step, lora_lr, lora_batch_size, lora_rank


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
            # Same fixed `height` on both input_image and output_image (below) so the two
            # boxes always line up at the same height regardless of image content, while
            # deliberately NOT setting `width` so each box just fills its column (roughly
            # half the app width via the two-Column layout) instead of being forced square.
            # Gradio scales the image to fit within that box (object-fit: contain),
            # preserving the original aspect ratio rather than stretching/cropping it.
            input_image = gr.Image(type="numpy", label="Click on the image to mark points",
                                show_label=True, height=IMAGE_BOX_HEIGHT)  # for points clicking
            with gr.Row():
                undo_point_button = gr.Button("Undo Point")
                undo_pair_button = gr.Button("Undo Pair")
                
                #save_button = gr.Button('Save Current Data')
                #data_dir = gr.Textbox(value='./dataset/test', label="Data Directory",
                #                      placeholder="Enter directory path for mask and points")
            

        with gr.Column():
            gr.Markdown("<h2 style='text-align: left;'>Dragged Image</h2>")
            output_image = gr.Image(type="numpy", label="View the editing results here",
                                    show_label=True, height=IMAGE_BOX_HEIGHT)  # see note above
            with gr.Row():
                run_button = gr.Button("Run")
                #clear_all_button = gr.Button("Clear All")
                #save_result = gr.Button("Save Result")
                #show_points = gr.Button("Show Points")
                #result_save_path = gr.Textbox(value='./result/test', label="Result Folder",
                #                            placeholder="Enter path to save the results")

    # Full-width row (not nested inside a Column) so the LoRA training progress bar,
    # which Gradio overlays on top of this component while `progress.tqdm()` is
    # running, spans the entire horizontal width of the app instead of just one
    # half of it. It stays empty/invisible when idle since we removed the old
    # separate "LoRA Training Status" textbox.
    with gr.Row():
        lora_progress_display = gr.HTML(value="")

    return input_image, undo_point_button, undo_pair_button, \
           output_image, run_button, lora_progress_display


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

def attach_input_image_event(input_image, selected_points, original_image):
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

    input_image.upload(
        fn=upload_store_img,
        inputs=[input_image],
        outputs=[original_image, selected_points]
    ).then(
        fn=refresh_input_image_display,
        inputs=[original_image],
        outputs=[input_image]
    )

    input_image.select(
        fn=get_points,
        inputs=[input_image, selected_points],
        outputs=[input_image]
    )

def attach_undo_button_event(undo_point_button, undo_pair_button, original_image, selected_points, input_image):
    undo_point_button.click(
        undo_point,
        [original_image, selected_points],
        [input_image, selected_points]
    )
    undo_pair_button.click(
        undo_pair,
        [original_image, selected_points],
        [input_image, selected_points]
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

previous_original_image = None


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
    global previous_original_image
    prompt = ""
    if previous_original_image is None or not np.array_equal(original_image, previous_original_image):
        train_lora_interface(original_image, prompt, model_path, vae_path,
                              lora_path, lora_step, lora_lr, lora_batch_size, lora_rank,
                              progress=progress)
    previous_original_image = original_image
    return ""


def run_direct_drag_step(original_image, input_image, selected_points,
         inversion_strength, lam, latent_lr, model_path, vae_path,
         lora_path, drag_end_step, drag_per_step, r1, r2, d,
         max_drag_per_track, max_track_no_change, feature_idx, save_intermediates_images,
         enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function,
         soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio):
    result_save_path = "gadio_results"
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
    # Return the filepath (not the raw array) so Gradio's built-in download icon on the
    # output Image component downloads using this filename instead of a generic one.
    return result_save_filename, new_points

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
                            lora_progress_display):
    run_button.click(
        run_lora_training_step,
        [original_image, model_path, vae_path,
         lora_path, lora_step, lora_lr, lora_batch_size, lora_rank],
        [lora_progress_display]
    ).then(
        run_direct_drag_step,
        [original_image, input_image, selected_points,
         inversion_strength, lam, latent_lr, model_path, vae_path,
         lora_path, drag_end_step, drag_per_step, r1, r2, d,
         max_drag_per_track, max_track_no_change, feature_idx, save_intermediates_images,
         enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function,
         soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio],
        [output_image, new_points]
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
    with gr.Blocks() as demo:
        selected_points = gr.State([])
        new_points = gr.State([])
        original_image = gr.State(value=None)
        create_markdown_section()
        intermediate_images = gr.State([])

        input_image, undo_point_button, undo_pair_button, \
        output_image, run_button, lora_progress_display = create_real_image_editing_ui()

        enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function, \
           soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio = create_directdrag_parameters_ui()
        
        latent_lr, drag_end_step, drag_per_step = create_drag_parameters_ui()

        model_path, vae_path = create_base_model_config_ui()
        lora_path, lora_step, lora_lr, lora_batch_size, lora_rank = create_lora_parameters_ui()
        r1, r2, d, feature_idx, max_drag_per_track, lam, inversion_strength, max_track_no_change = \
            create_advance_parameters_ui()
        save_intermediates_images, get_mp4_button = create_intermediate_save_ui()

        attach_input_image_event(input_image, selected_points, original_image)
        attach_undo_button_event(undo_point_button, undo_pair_button, original_image, selected_points, input_image)
        attach_run_button_event(run_button, original_image, input_image, selected_points,
                                inversion_strength, lam, latent_lr, model_path, vae_path, 
                                lora_path, lora_step, lora_lr, lora_batch_size, lora_rank,
                                drag_end_step, drag_per_step, output_image,
                                r1, r2, d, feature_idx, new_points, max_drag_per_track,
                                max_track_no_change, save_intermediates_images,
                                enable_soft_mask, enable_readout_guided_feature_alignment, enable_latent_warpage_function,
                                soft_mask_sigma, readout_guided_feature_alignment_multiplier, latent_warpage_function_ratio,
                                lora_progress_display)
        #attach_show_points_event(show_points, output_image, new_points)
        #attach_clear_all_button_event(clear_all_button, input_image, output_image, selected_points,
        #                              original_image)

    demo.queue().launch(share=False, debug=True)


if __name__ == '__main__':
    main()