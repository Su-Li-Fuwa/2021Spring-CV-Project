# python 3.7
"""Demo."""

import numpy as np
import torch
import streamlit as st
import SessionState
import pandas as pd
import os, cv2
import urllib.request as urlreq
from main import directionFindByInversion_modified
from main import manipulation_modified
from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image
from utils.editor import manipulate
from PIL import Image
import datetime

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
            with urlreq.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
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
    original_image, modified_image, direction, magnitude_d, step_num, img_type, real_img, semantic_type = selector_ui(image_size)
    
    if original_image.any(): draw_image(original_image, modified_image)

    # debugged!
    @st.cache(hash_funcs={StyleGANInverter: lambda _: None})
    def find_direction(original_image, modified_image, inverter):
        return directionFindByInversion_modified(original_image, modified_image, inverter)

    @st.cache(hash_funcs={StyleGANInverter: lambda _: None})
    def find_latent_code(real_img, inverter):
        return inverter.easy_invert(real_img, 0)[0]
    
    real_img_code = []
    if img_type == "upload":
        real_img_code = [find_latent_code(real_img[idx], inverter)[0] for idx in range(len(real_img)) ]

    dire = 0
    for idx in range(0, len(direction)):
        if (direction[idx] == -1).all():
            dire += find_direction(original_image, modified_image, inverter) * magnitude_d[idx]
        else:
            dire += direction[idx] * magnitude_d[idx]

    manipulate_layers = list(range(inverter.G.num_layers))

    if st.button("Run"):
        if img_type == "upload":
            target = manipulation_modified(inverter, dire, manipulate_layers, -1, 1, step_num, image_path = None, real_img_code= np.array(real_img_code))
        else:
            target = manipulation_modified(inverter, dire, manipulate_layers, -1, 1, step_num)
        draw_image_series(target, image_size)
    
    try:
        direction_name = st.text_input("Direction name: ") #, "DIY_" + str(datetime.datetime.now()))
        if(direction_name):
            if st.button("Save"):
                save_path = f'data/{direction_name}/'
                dire_list = []
                dire_list.append(dire)
                if not os.path.exists(f'data/{direction_name}'):
                    os.mkdir(f'data/{direction_name}')
                np.savez(save_path+'direction.npz', direction = np.array(dire_list))      
                cv2.imwrite(save_path + 'ori.png', cv2.cvtColor(np.asarray(original_image),cv2.COLOR_BGR2RGB))
                cv2.imwrite(save_path + 'mod.png', cv2.cvtColor(np.asarray(modified_image),cv2.COLOR_BGR2RGB))
                with open(f'data/semantic_list.txt', 'a') as f:
                    f.write(direction_name+'\n')
                st.success("Saved the direction and the images to ./data/"+direction_name+"!")
    except:
        st.warning("You should only save 'DIY' directions. Check whether the semantic is 'DIY'.")

def selector_ui(image_size):
    st.sidebar.markdown("# Settings")

    num_semantics = 3
    magnitude_d = []
    direction = []
    original_image, modified_image, real_img = np.array([]), np.array([]), np.array([])

    f = open("data/semantic_list.txt")
    sems = f.read().splitlines()
    f.close()
    for name in sems:
        assert os.path.exists(f'data/{name}/ori.png')
        assert os.path.exists(f'data/{name}/mod.png')
        assert os.path.exists(f'data/{name}/direction.npz')
    sems.insert(0, "None")
    sems.append("DIY")

    for sem_idx in range(num_semantics):

        semantic_type = st.sidebar.selectbox(f"Choose semantics {sem_idx}: ", sems, 0)
        if semantic_type == "None": continue
        magnitude_d.append(st.sidebar.slider(f"Set the magnitude of {sem_idx}: ", 0.0, 3.0, 0.5))

        if semantic_type == "DIY":
            ori = st.file_uploader("Original Image here: ",type=['png','jpeg','jpg'])
            mod = st.file_uploader("Self-modified Image here: ",type=['png','jpeg','jpg'])

            if np.array(mod == None).any(): continue

            ori = Image.open(ori)
            mod = Image.open(mod)
            ori = cv2.cvtColor(np.asarray(ori),cv2.COLOR_RGB2BGR)
            mod = cv2.cvtColor(np.asarray(mod),cv2.COLOR_RGB2BGR)
            original_image = ori[:, :, ::-1]
            modified_image = mod[:, :, ::-1]
            original_image = resize_image(original_image, (image_size, image_size))
            modified_image = resize_image(modified_image, (image_size, image_size))
            direction.append(np.array([-1]))    # currently only allow one DIY, direction needs to be calculated.
        
        else:
            try:
                # original_image = load_image(f'data/{semantic_type}/ori.png')
                # modified_image = load_image(f'data/{semantic_type}/mod.png')
                # if os.path.exists(f'data/{semantic_type}/direction.npz'):
                direction.append(np.load(f'data/{semantic_type}/direction.npz')['direction'].mean(axis = 0))
                # else: direction.append(np.array([-1]))
            except FileNotFoundError:
                pass   # predetermined feature needs saved direction. 
    
    step_n = st.sidebar.slider(f"Set the number of step: ", 1, 4, 1, 1)
    
    # whether use default image set
    img_type = st.sidebar.selectbox(f"Choose image set: ", ["default", "upload"], 0)
    real_img_list = []
    if img_type == "upload":
        file_list = st.file_uploader(f"Real image here: ", type=['png','jpeg','jpg'], accept_multiple_files=True)
        for img in file_list:
            real_img = Image.open(img)
            real_img = cv2.cvtColor(np.asarray(real_img),cv2.COLOR_RGB2BGR)
            real_img = real_img[:, :, ::-1]
            real_img = resize_image(real_img, (image_size, image_size))
            real_img_list.append(real_img)

    return original_image, modified_image, direction, magnitude_d, step_n, img_type, real_img_list, semantic_type #, image_idx

def draw_image(img1, img2):
    st.subheader("Original Image / Manually Modified Image")
    st.image([img1,img2])

def draw_image_series(imgs, image_size):
    st.subheader("Results")
    for img_series in imgs:
        st.image(img_series, width=int(image_size*2/len(img_series)))

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "./models/pretrain/styleganinv_ffhq256_generator.pth": {
        "url": "https://cloud.tsinghua.edu.cn/d/15e59c417fd34fad95f0/files/?p=%2Fstyleganinv_ffhq256_generator.pth&dl=1",
        "size": 118807232
    },
    "./models/pretrain/styleganinv_ffhq256_encoder.pth": {
        "url": "https://cloud.tsinghua.edu.cn/d/15e59c417fd34fad95f0/files/?p=%2Fstyleganinv_ffhq256_encoder.pth&dl=1",
        "size": 661456905
    },
    "./models/pretrain/vgg16.pth": {
        "url": "https://cloud.tsinghua.edu.cn/d/15e59c417fd34fad95f0/files/?p=%2Fvgg16.pth&dl=1",
        "size": 58862227
    }
}
if __name__ == '__main__':
    main()
