import cv2
import os
import platform
import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms
from modules import script_callbacks, devices, shared
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler
from file_manager import file_manager as file_manager_lama


def get_model_ids():
    """Get cleaner model ids list.

    Returns:
        list: model ids list
    """
    model_ids = [
        "lama",
        "ldm",
        "zits",
        "mat",
        "fcf",
        "manga",
    ]
    return model_ids


def auto_resize_to_pil(input_image, mask_image):
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size, "The sizes of the image and mask do not match"
    width, height = init_image.size

    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        resize_height = int(height * scale + 0.5)
        resize_width = int(width * scale + 0.5)
        if height != resize_height or width != resize_width:
            # ia_logging.info(f"resize: ({height}, {width}) -> ({resize_height}, {resize_width})")
            init_image = transforms.functional.resize(init_image, (resize_height, resize_width),
                                                      transforms.InterpolationMode.LANCZOS)
            mask_image = transforms.functional.resize(mask_image, (resize_height, resize_width),
                                                      transforms.InterpolationMode.LANCZOS)
        if resize_height != new_height or resize_width != new_width:
            # ia_logging.info(f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})")
            init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
            mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))

    return init_image, mask_image


def run_lama(inputs, model_id):
    input_image = inputs["image"]
    input_mask = inputs["mask"]
    if platform.system() == "Darwin":
        model = ModelManager(name=model_id, device=devices.cpu)
    else:
        model = ModelManager(name=model_id, device=devices.device)

    init_image, mask_image = auto_resize_to_pil(input_image, input_mask)
    width, height = init_image.size

    init_image = np.array(init_image)
    mask_image = np.array(mask_image.convert("L"))

    config = Config(
        ldm_steps=20,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=512,
        prompt="",
        sd_steps=20,
        sd_sampler=SDSampler.ddim
    )

    output_image = model(image=init_image, mask=mask_image, config=config)
    output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_image)

    save_name = "_".join([file_manager_lama.savename_prefix, os.path.basename(model_id)]) + ".png"
    save_name = os.path.join(file_manager_lama.outputs_dir, save_name)
    output_image.save(save_name)
    return [output_image]


def on_ui_tabs():
    model_ids = get_model_ids()
    out_gallery_kwargs = dict(columns=2, height=520, object_fit="contain", preview=True)

    with gr.Blocks(analytics_enabled=False) as lama_interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    inputs = gr.Image(tool="sketch", label="Input", type="numpy")

                with gr.Tab("lama", elem_id='lama_tab'):
                    with gr.Row():
                        with gr.Column():
                            model_id = gr.Dropdown(label="Model ID", elem_id="model_id", choices=model_ids,
                                                   value=model_ids[0], show_label=True)
                        with gr.Column():
                            lama_btn = gr.Button("Run lama", elem_id="lama_btn", variant="primary")

            with gr.Column():
                output = gr.Gallery(label="cleaned_image", elem_id="cleaner_output", show_label=False).style(
                    **out_gallery_kwargs)

        lama_btn.click(run_lama, inputs=[inputs, model_id], outputs=[output])

    return [(lama_interface, "lama", "Lama_Cleaner")]


script_callbacks.on_ui_tabs(on_ui_tabs)
