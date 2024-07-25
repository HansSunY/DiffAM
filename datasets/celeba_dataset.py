import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from io import BytesIO
import PIL
from PIL import Image
import torchvision.transforms as tfs
import torchvision.utils as tvu

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, mark,resolution=256):
        self.path = path
        self.attr_path = os.path.join(self.path, "CelebAMask-HQ-attribute-anno.txt")
        self.img_path = os.path.join(self.path,"CelebA-HQ-img")
        self.resolution = resolution
        self.transform = transform
        self.transform_mask = tfs.Compose([
        tfs.Resize((256,256),interpolation=PIL.Image.NEAREST),
        tfs.ToTensor()])
        self.dataset = []
        self.preprocess(mark)
        
    def preprocess(self,mark):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        if mark == 0:
            lines = lines[4000:5000]
        else:
            lines = lines[15000:16000]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            self.dataset.append(filename)

        print('Finished preprocessing the CelebA dataset...')
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        filename = self.dataset[index]
        image = Image.open(os.path.join(self.img_path, filename))
        id = int(filename[:-4])//2000
        maskname = filename[:-4].zfill(5)
        mask_root = os.path.join(self.path, "CelebAMask-HQ-mask-anno/{}/".format(id))
        l_eye_path = os.path.join(mask_root, maskname+"_l_eye.png")
        r_eye_path = os.path.join(mask_root, maskname+"_r_eye.png")
        l_brow_path = os.path.join(mask_root, maskname+"_l_brow.png")
        r_brow_path = os.path.join(mask_root, maskname+"_r_brow.png")
        l_lip_path = os.path.join(mask_root, maskname+"_l_lip.png")
        u_lip_path = os.path.join(mask_root, maskname+"_u_lip.png")
        skin_path = os.path.join(mask_root, maskname+"_skin.png")
        mouth_path = os.path.join(mask_root, maskname+"_mouth.png")
        neck_path = os.path.join(mask_root, maskname+"_neck.png")
        if not os.path.exists(l_eye_path) or not os.path.exists(r_eye_path) or not os.path.exists(l_lip_path) or not os.path.exists(u_lip_path):
            mask_list = 0
        else:
            image_l_eye  = self.transform_mask(Image.open(l_eye_path))
            image_r_eye  = self.transform_mask(Image.open(r_eye_path))
            image_l_lip  = self.transform_mask(Image.open(l_lip_path))
            image_u_lip  = self.transform_mask(Image.open(u_lip_path))
            image_skin  = self.transform_mask(Image.open(skin_path))
            image_lip = torch.clamp(image_u_lip + image_l_lip,0,1)
            image_face = image_skin - image_l_eye - image_r_eye - image_lip
            image_skin = image_skin - image_l_eye - image_r_eye - image_lip
            if os.path.exists(l_brow_path):
                image_l_brow = self.transform_mask(Image.open(l_brow_path))
                image_skin = image_skin-image_l_brow
                image_face = image_face-image_l_brow
            if os.path.exists(r_brow_path):
                image_r_brow = self.transform_mask(Image.open(r_brow_path))
                image_skin = image_skin-image_r_brow
                image_face = image_face-image_r_brow
            if os.path.exists(neck_path):
                image_neck = self.transform_mask(Image.open(neck_path))
                image_skin = image_skin+image_neck
            if os.path.exists(mouth_path):
                image_mouth = self.transform_mask(Image.open(mouth_path))
                image_skin = image_skin-image_mouth
                image_face = image_face-image_mouth
            mask_list = [image_l_eye,image_r_eye,image_lip,torch.clamp(image_skin,0,1),torch.clamp(image_face,0,1)]
        return self.transform(image), mask_list


################################################################################

def get_celeba_dataset(data_root, config):
    transform = tfs.Compose([tfs.Resize(256),tfs.ToTensor(),tfs.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    train_dataset = MultiResolutionDataset(data_root,
                                           transform, 0,config.data.image_size)
    test_dataset = MultiResolutionDataset(data_root,
                                          transform, 1,config.data.image_size)


    return train_dataset, test_dataset