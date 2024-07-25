from torch.utils.data import Dataset
import torchvision.transforms as tfs
import os
from utils.align_utils import *
from PIL import Image
from utils.image_processing import ToTensor


class MultiResolutionDataset(Dataset):
    def __init__(self, mark, path, transform, resolution=256):
        self.images = os.listdir(path + 'images/makeup/')

        if mark == 0:
            self.train_paths = [path + 'images/makeup/' +
                                _ for _ in self.images][500:]
            self.mask_paths = [path + 'segs/makeup/' +
                               _ for _ in self.images][500:]
        else:
            self.train_paths = [path + 'images/makeup/' +
                                _ for _ in self.images][:500]
            self.mask_paths = [path + 'segs/makeup/' +
                               _ for _ in self.images][:500]
                               
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, index):
        img_name = self.train_paths[index]
        mask_name = self.mask_paths[index]

        aligned_image = Image.open(img_name).resize(
            (self.resolution, self.resolution))
        mask = Image.open(mask_name)

        img = self.transform(aligned_image)
        return img, ToTensor(mask)


################################################################################

def get_mt_dataset(data_root, config):
    transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

    train_dataset = MultiResolutionDataset(
        0, data_root, transform, config.data.image_size)
    test_dataset = MultiResolutionDataset(
        1, data_root, transform, config.data.image_size)

    return train_dataset, test_dataset
