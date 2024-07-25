from torch.utils.data import DataLoader
from .celeba_dataset import get_celeba_dataset
from .mt_dataset import get_mt_dataset
def get_dataset(dataset_type, dataset_paths, config):
    if dataset_type == 'CelebA_HQ':
        train_dataset, test_dataset = get_celeba_dataset(dataset_paths['CelebA_HQ'], config)
    elif dataset_type == 'MT':
        train_dataset, test_dataset = get_mt_dataset(dataset_paths['MT'],config)
    else:
        raise ValueError    
    return train_dataset, test_dataset

def get_dataloader(train_dataset, test_dataset, bs_train=1, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs_train,
        drop_last=True,
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True,
        sampler=None,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {'train': train_loader, 'test': test_loader}


