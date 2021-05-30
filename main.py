import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image
from utils.editor import manipulate


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of the GAN model.')
    parser.add_argument('real_img_path', type=str,
                        help='path to the real images')
    parser.add_argument('-f', '--feature_name', type=str,
                        help='feature name')
    
    # parameters for inversion
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimization. (default: 0.01)')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of optimization iterations. (default: 100)')
    parser.add_argument('--num_results', type=int, default=5,
                        help='Number of intermediate optimization results to '
                            'save for each sample. (default: 5)')
    parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                        help='The perceptual loss scale for optimization. '
                            '(default: 5e-5)')
    parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                        help='The encoder loss scale for optimization.'
                            '(default: 2.0)')
    
    # parameters for manipulation
    parser.add_argument('--step', type=int, default=7,
                        help='Number of manipulation steps. (default: 7)')
    parser.add_argument('--start_distance', type=float, default=-1.0,
                        help='Start distance for manipulation. (default: -1.0)')
    parser.add_argument('--end_distance', type=float, default=1.0,
                        help='End distance for manipulation. (default: 1.0)')
    
    parser.add_argument('--viz_size', type=int, default=256,
                        help='Image size for visualization. (default: 256)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')
    return parser.parse_args()

def directionFindByInversion_modified(ori_img, mod_img, inverter):
    #image_size = inverter.G.resolution
    direction_list = []
    ori_code, _ = inverter.easy_invert(ori_img, 5)
    mod_code, _ = inverter.easy_invert(mod_img, 5)
    direction_list.append(mod_code-ori_code)
    # if is_save:
    #     feature_name = src_path.split('/')[-1]
    #     save_path = f'results/{feature_name}/direction.npz'
    #     np.savez(save_path, ori_code = np.array(ori_code_list), mod_code = np.array(mod_code_list),
    #                             direction = np.array(direction_list))
    return np.array(direction_list).mean(axis = 0)


def directionFindByInversion(src_path, inverter, is_save = True):
    """
    Find specified direction
    """
    image_size = inverter.G.resolution
    feature_name = src_path.split('/')[-1]
    ori_path, mod_path = os.path.join(src_path, "original"), os.path.join(src_path, "modified")
    assert os.path.exists(ori_path) and os.path.exists(mod_path) , "Path Error"

    ori_list, mod_list = list(os.walk(ori_path))[0][2], list(os.walk(mod_path))[0][2]
    assert len(ori_list) == len(mod_list)

    ori_code_list, mod_code_list, direction_list = [], [], []

    for ori_img_path, mod_img_path in zip(ori_list, mod_list):
        ori_img_path = os.path.join(src_path, "original", ori_img_path)
        mod_img_path = os.path.join(src_path, "modified", mod_img_path)
        ori_img = resize_image(load_image(ori_img_path), (image_size, image_size))
        mod_img = resize_image(load_image(mod_img_path), (image_size, image_size))
        ori_code, _ = inverter.easy_invert(ori_img, 5)
        mod_code, _ = inverter.easy_invert(mod_img, 5)
        ori_code_list.append(ori_code), mod_code_list.append(mod_code)
        direction_list.append(mod_code-ori_code)
    
    if is_save:
        feature_name = src_path.split('/')[-1]
        save_path = f'results/{feature_name}/direction.npz'
        np.savez(save_path, ori_code = np.array(ori_code_list), mod_code = np.array(mod_code_list),
                                direction = np.array(direction_list))
    
    return np.array(direction_list).mean(axis = 0)

def manipulation_modified(inverter, direction, layers, start_distance, end_distance, step, image_path = 'data/real_image', real_img_code = None):
    """ do the manipulation with given direction and latent code """
    generator = inverter.G
    if image_path:
        image_dir = image_path
        assert os.path.exists(image_dir)
        assert os.path.exists(f'{image_dir}/inverted_codes.npy')    # Simplified

        image_list = []
        with open(f'{image_dir}/image_list.txt', 'r') as f:
            for line in f:
                name = os.path.splitext(os.path.basename(line.strip()))[0]
                assert os.path.exists(f'{image_dir}/{name}_ori.png')
                assert os.path.exists(f'{image_dir}/{name}_inv.png')
                image_list.append(name)
        latent_codes = np.load(f'{image_dir}/inverted_codes.npy')
        assert latent_codes.shape[0] == len(image_list)

    else:
        latent_codes = real_img_code
    #print(latent_codes.shape)
    num_images = latent_codes.shape[0]

    boundary = direction
    manipulate_layers = layers

    codes = manipulate(latent_codes=latent_codes,
                        boundary=boundary,
                        start_distance=start_distance,
                        end_distance=end_distance,
                        step=step,
                        layerwise_manipulation=True,
                        num_layers=generator.num_layers,
                        manipulate_layers=manipulate_layers,
                        is_code_layerwise=True,
                        is_boundary_layerwise=True)
    results = []
    for img_idx in tqdm(range(num_images), leave=False):
        output_images = generator.easy_synthesize(codes[img_idx], latent_space_type='wp')['image']
        results.append(output_images)

    return results


def manipulation(image_path, inverter, direction, layers, step = 7, viz_size = 256, 
                 start_distance = -3., end_distance = 3., feature_name = 'temp'):
    """ do the manipulation with given direction and latent code """
    image_dir = image_path
    image_dir_name = os.path.basename(image_dir.rstrip('/'))
    assert os.path.exists(image_dir)
    assert os.path.exists(f'{image_dir}/inverted_codes.npy')    # Simplified

    output_dir = 'results/' + feature_name 

    generator = inverter.G

    image_list = []
    with open(f'{image_dir}/image_list.txt', 'r') as f:
        for line in f:
            name = os.path.splitext(os.path.basename(line.strip()))[0]
            assert os.path.exists(f'{image_dir}/{name}_ori.png')
            assert os.path.exists(f'{image_dir}/{name}_inv.png')
            image_list.append(name)
    latent_codes = np.load(f'{image_dir}/inverted_codes.npy')
    assert latent_codes.shape[0] == len(image_list)
    num_images = latent_codes.shape[0]


    boundary = direction
    manipulate_layers = layers

    # Manipulate images.
    visualizer = HtmlPageVisualizer(
        num_rows=num_images, num_cols=step + 3, viz_size=viz_size)
    visualizer.set_headers(
        ['Name', 'Origin', 'Inverted'] +
        [f'Step {i:02d}' for i in range(1, step + 1)]
    )
    for img_idx, img_name in enumerate(image_list):
        ori_image = load_image(f'{image_dir}/{img_name}_ori.png')
        inv_image = load_image(f'{image_dir}/{img_name}_inv.png')
        visualizer.set_cell(img_idx, 0, text=img_name)
        visualizer.set_cell(img_idx, 1, image=ori_image)
        visualizer.set_cell(img_idx, 2, image=inv_image)

    codes = manipulate(latent_codes=latent_codes,
                        boundary=boundary,
                        start_distance=start_distance,
                        end_distance=end_distance,
                        step=step,
                        layerwise_manipulation=True,
                        num_layers=generator.num_layers,
                        manipulate_layers=manipulate_layers,
                        is_code_layerwise=True,
                        is_boundary_layerwise=True)

    for img_idx in tqdm(range(num_images), leave=False):
        output_images = generator.easy_synthesize(codes[img_idx], latent_space_type='wp')['image']
        for s, output_image in enumerate(output_images):
            visualizer.set_cell(img_idx, s + 3, image=output_image)

    # Save results.
    visualizer.save(f'{output_dir}/visualization.html')


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(f'results/{args.feature_name}'):
        os.mkdir(f'results/{args.feature_name}')

    inverter = StyleGANInverter(
                    args.model_name,
                    learning_rate=args.learning_rate,
                    iteration=args.num_iterations,
                    reconstruction_loss_weight=1.0,
                    perceptual_loss_weight=args.loss_weight_feat,
                    regularization_loss_weight=args.loss_weight_enc,
                    logger=None)
    
    if os.path.exists(f'results/{args.feature_name}/direction.npz'):
        direction = np.load(f'results/{args.feature_name}/direction.npz')['direction'].mean(axis = 0)
    else:   direction = directionFindByInversion(f'data/{args.feature_name}', inverter, True)

    manipulate_layers = list(range(inverter.G.num_layers))
    manipulation(args.real_img_path, inverter, direction, 
                manipulate_layers, 
                args.step,
                args.viz_size,
                args.start_distance,
                args.end_distance,
                args.feature_name)