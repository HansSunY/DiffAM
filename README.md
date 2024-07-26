# DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection (CVPR 2024)

[![arXiv](https://img.shields.io/badge/paper-cvpr2024-cyan)](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_DiffAM_Diffusion-based_Adversarial_Makeup_Transfer_for_Facial_Privacy_Protection_CVPR_2024_paper.html) [![arXiv](https://img.shields.io/badge/arXiv-2405.09882-red)](https://arxiv.org/abs/2405.09882)

Official PyTorch  implementation of paper "DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection".

## Abstract

With the rapid development of face recognition (FR) sys tems, the privacy of face images on social media is facing severe challenges due to the abuse of unauthorized FR sys tems. Some studies utilize adversarial attack techniques to defend against malicious FR systems by generating adversarial examples. However, the generated adversarial examples, i.e., the protected face images, tend to suffer from sub par visual quality and low transferability. In this paper, we propose a novel face protection approach, dubbed DiffAM, which leverages the powerful generative ability of diffusion models to generate high-quality protected face images with adversarial makeup transferred from reference images. To be specific, we first introduce a makeup removal module to generate non-makeup images utilizing a fine-tuned diffusion model with guidance of textual prompts in CLIP space. As the inverse process of makeup transfer, makeup removal can make it easier to establish the deterministic relation ship between makeup domain and non-makeup domain regardless of elaborate text prompts. Then, with this relationship, a CLIP-based makeup loss along with an ensemble attack strategy is introduced to jointly guide the direction of adversarial makeup domain, achieving the generation of protected face images with natural-looking makeup and high black-box transferability. Extensive experiments demonstrate that DiffAM achieves higher visual quality and attack success rates with a gain of 12.98% under black-box setting compared with the state of the arts.

## Setup

- ### Get code

```shell
git clone https://github.com/HansSunY/DiffAM.git
```

- ### Build environment

```shell
cd DiffAM
# use anaconda to build environment 
conda create -n diffam python=3.8
conda activate diffam
# install packages
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## Pretrained models and datasets

- The weights required for the execution of DiffAM can be downloaded [here](https://drive.google.com/drive/folders/1L8caY-FVzp9razKMuAt37jCcgYh3fjVU?usp=sharing). 

```shell
mkdir pretrained
mv celeba_hq.ckpt pretrained/
mv makeup.pt pretrained/
mv model_ir_se50.pth pretrained/
mv shape_predictor_68_face_landmarks.dat pretrained/
```

- Please download the target FR models, MT-datasets and target images [here](https://drive.google.com/file/d/1IKiWLv99eUbv3llpj-dOegF3O7FWW29J/view?usp=sharing). Unzip the assets.zip file in `DiffAM/assets`.
- Please download the [CelebAMask-HQ](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) dataset and unzip the file in `DiffAM/assets/datasets`.

The final project should be like this:

```shell
DiffAM
  └- assets
     └- datasets
     	└- CelebAMask-HQ
     	└- MT-dataset
     	└- pairs
     	└- target
     	└- test
     └- models
  └- pretrained
       └- celeba_hq.ckpt
       └- ...
  └- ...
```

## Quick Start

### Makeup removal (Optional)

- We have included five makeup styles for adversarial makeup transfer in `DiffAM/assets/datasets/pairs`, which comprises pairs of makeup and non-makeup images, along with their corresponding masks. Therefore, you can directly **skip** the step and proceed to try out makeup transfer using the provided styles.
- If you want to fine-tune the pretrained diffusion model for makeup removal and generate more pairs of makeup and non-makeup images, please run the following commands:

```shell
python main.py --makeup_removal --config MT.yml --exp ./runs/test --do_train 1 --do_test 1 --n_train_img 200 --n_test_img 100 --n_iter 7 --t_0 300 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --model_path pretrained/makeup.pt
```

Then you can remove the makeup with the trained model and put the pairs of makeup and non-makeup images along with corresponding masks in `DiffAM/assets/datasets/pairs` for the following adversarial makeup transfer.

### Adversarial makeup transfer

To fine-tuned the pretrained diffusion model for adversarial makeup transfer, please run the following commands:

```shell
python main.py --makeup_transfer --config celeba.yml --exp ./runs/test --do_train 1 --do_test 1 --n_train_img 200 --n_test_img 100 --n_iter 4 --t_0 60 --n_inv_step 20 --n_train_step 6 --n_test_step 6 --lr_clip_finetune 8e-6 --model_path pretrained/celeba_hq.ckpt --target_img 1 --target_model 2 --ref_img 'XMY-060'
```

- `target_img`: Choose the target identity to attack, a total of 4 options are provided (see details in our supplementary materials).

- `target_model`: Choose the target FR model to attack, including `[IRSE50, IR152, Mobileface, Facenet]`.
- `ref_img`: Choose the provided makeup style to transfer, including `['XMY-060', 'XYH-045', 'XMY-254', 'vRX912', 'vFG137']`. In addition, by generating pairs of makeup and non-makeup images through makeup removal, you can also transfer the makeup style you want. (Save `{ref_name}_m.png`, `{ref_name}_nm.png`, and `{ref_name}_mask.png` to `DiffAM/assets/datasets/pairs`.)

### Edit one image

You can edit one image for makeup removal and transfer by running the following command:

```shell
# makeup removal
python main.py --edit_one_image_MR --config MT.yml --exp ./runs/test --n_iter 1 --t_0 300 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --img_path {IMG_PATH} --model_path {MODEL_PATH}

# adversarial makeup removal
python main.py --edit_one_image_MT --config celeba.yml --exp ./runs/test --n_iter 1 --t_0 60 --n_inv_step 20 --n_train_step 6 --n_test_step 6 --img_path {IMG_PATH} --model_path {MODEL_PATH}
```

- `img_path`: Path of an image to edit.
- `model_path`: Path of fine-tuned model.

## Citation

```bibtex
@InProceedings{Sun_2024_CVPR,
    author    = {Sun, Yuhao and Yu, Lingyun and Xie, Hongtao and Li, Jiaming and Zhang, Yongdong},
    title     = {DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {24584-24594}
}
```

## Acknowledgments

Our code structure is based on [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP?tab=readme-ov-file) and [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN).
