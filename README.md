# Shared Repo. for 2021 Spring Computer Vision Project

The codes are mainly borrowed from [idinvert_pytorch](https://github.com/genforce/idinvert) and integrated in `main.py`

## Specified direction identification

Please download the pre-trained models from the following links and save them to `models/pretrain/`

| Description | Generator | Encoder |
| :---------- | :-------- | :------ |
| Model trained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. | [face_256x256_generator](https://drive.google.com/file/d/1SjWD4slw612z2cXa3-n38JwKZXqDUerG/view?usp=sharing)    | [face_256x256_encoder](https://drive.google.com/file/d/1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO/view?usp=sharing)
| Model trained on [LSUN Tower](https://github.com/fyu/lsun) dataset.      | [tower_256x256_generator](https://drive.google.com/file/d/1lI_OA_aN4-O3mXEPQ1Nv-6tdg_3UWcyN/view?usp=sharing)   | [tower_256x256_encoder](https://drive.google.com/file/d/1Pzkgdi3xctdsCZa9lcb7dziA_UMIswyS/view?usp=sharing)
| Model trained on [LSUN Bedroom](https://github.com/fyu/lsun) dataset.    | [bedroom_256x256_generator](https://drive.google.com/file/d/1ka583QwvMOtcFZJcu29ee8ykZdyOCcMS/view?usp=sharing) | [bedroom_256x256_encoder](https://drive.google.com/file/d/1ebuiaQ7xI99a6ZrHbxzGApEFCu0h0X2s/view?usp=sharing)
| [Perceptual Model](https://drive.google.com/file/d/1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y/view?usp=sharing)

For a specified direction, please store the original images in `data/$feature_name/original` and corresponding modified images in `data/$feature_name/modified`. The inversion results of real images are already saved in `data/real_image`

```bash
python3 main.py $model_name $real_image_path -f $feature_name
```

`main.py` will first do the GAN inversion on all original and modified images, get the latent codes and do the subtraction to get the specified direction (If more than one instances are given, do the average). It will save all latent codes and the direction in `results/$feature_name/direction.npz`, then apply this direction to all real images. The visualization will be saved in `results/$feature_name/visualization.html`


** A simple example: **
```bash
python3 main.py 'styleganinv_ffhq256' 'data/real_image' -f glasses
```