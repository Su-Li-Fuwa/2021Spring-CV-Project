# python 3.7
"""Demo."""

import numpy as np
import torch
import streamlit as st
import SessionState
import pandas as pd
import os, urllib, cv2
from main import directionFindByInversion_modified
from main import manipulation_modified
from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image
from utils.editor import manipulate
from PIL import Image

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    with open('instructions.md', 'r') as f:
        instructions = f.read()
    readme_text = st.markdown(instructions)

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()

# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    @st.cache(allow_output_mutation=True)
    def load_model():
        inverter = StyleGANInverter(
                        model_name='styleganinv_ffhq256',
                        learning_rate=0.01,
                        iteration=100,
                        reconstruction_loss_weight=1.0,
                        perceptual_loss_weight=5e-5,
                        regularization_loss_weight=2.0,
                        logger=None)
        return inverter
    inverter = load_model()
    image_size = inverter.G.resolution

    # original_image, modified_image, direction, start_d, end_d, step_l, image_idx = selector_ui(image_size)
    original_image, modified_image, direction, start_d, end_d, step_l = selector_ui(image_size)
    draw_image(original_image, modified_image)

    # debugged!
    @st.cache(hash_funcs={StyleGANInverter: lambda _: None})
    def find_direction(original_image, modified_image, inverter):
        return directionFindByInversion_modified(original_image, modified_image, inverter)


    if direction is None:
        dire = find_direction(original_image, modified_image, inverter)
    else:   dire = direction

    manipulate_layers = list(range(inverter.G.num_layers))
    step_num = int((end_d - start_d)/step_l)

    if st.button("Run"):
        target = manipulation_modified(inverter, dire, manipulate_layers, start_d, end_d, step_num)
        draw_image_series(target)

def selector_ui(image_size):
    st.sidebar.markdown("# Settings")

    semantic_type = st.sidebar.selectbox("Choose semantics: ", ["glasses", "DIY"], 0)
    start_d, end_d = st.sidebar.slider("Set the step length: ", -3.0, 3.0, [-0.5,0.5])
    step_l = st.sidebar.slider("Set the step length: ", 0.0, 2.0, 0.25)
    # image_idx = st.sidebar.number_input("Image index: ", value=10, min_value=0, max_value=None, step=1)

    # if not os.path.exists(f'results/{semantic_type}'):
    #     os.mkdir(f'results/{semantic_type}')

    if semantic_type == "DIY":
        ori = st.file_uploader("Original Image here: ",type=['png','jpeg','jpg'])
        mod = st.file_uploader("Self-modified Image here: ",type=['png','jpeg','jpg'])
        ori = Image.open(ori)
        mod = Image.open(mod)
        ori = cv2.cvtColor(np.asarray(ori),cv2.COLOR_RGB2BGR)
        mod = cv2.cvtColor(np.asarray(mod),cv2.COLOR_RGB2BGR)
        original_image = ori[:, :, ::-1]
        modified_image = mod[:, :, ::-1]
        direction = None

    else:
        try:
            original_image = load_image(f'data/{semantic_type}/ori.png')
            modified_image = load_image(f'data/{semantic_type}/mod.png')
            if os.path.exists(f'results/{semantic_type}/direction.npz'):
                direction = np.load(f'results/{semantic_type}/direction.npz')['direction'].mean(axis = 0)
            else: direction = None 

        except FileNotFoundError:
            pass   # ?????
            
    original_image = resize_image(original_image, (image_size, image_size))
    modified_image = resize_image(modified_image, (image_size, image_size))

    return original_image, modified_image, direction, start_d, end_d, step_l#, image_idx

def draw_image(img1, img2):
    st.subheader("Original Image / Manually Modified Image")
    st.image([img1,img2])

def draw_image_series(imgs):
    st.subheader("Results")
    for img_series in imgs:
        st.image(img_series)

# External files to download.
EXTERNAL_DEPENDENCIES = {
    # "yolov3.weights": {
    #     "url": "https://pjreddie.com/media/files/yolov3.weights",
    #     "size": 248007048
    # },
    # "yolov3.cfg": {
    #     "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    #     "size": 8342
    # }
}
if __name__ == '__main__':
    main()
