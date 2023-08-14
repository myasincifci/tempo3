from torch.utils.data import DataLoader
from torchvision import transforms as T

from tempo3.data.video_dataset import VideoDataset
from tempo3.data.video_dataset_h5 import VideoDatasetH5
from tempo3.data.finetune_dataset import FinetuneDataset

transform = T.Compose([
    T.Resize(128),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def video_dataset(batch_size=80, proximity=30, pdf=None, num_workers=4):
    """
    Creates dataloader for tempo ss-training.
    """

    dataset = VideoDataset('./datasets/ASL-big/frames',
                           transform=transform, proximity=proximity, pdf=pdf)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=False, num_workers=num_workers)

    return dataloader

def video_dataset_h5(path: str, batch_size=80, proximity=30, pdf=None, num_workers=4):
    """
    Creates dataloader for tempo ss-training.
    """

    dataset = VideoDatasetH5(path,
                           transform=None, proximity=proximity, pdf=pdf)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=False, num_workers=num_workers)

    return dataloader

def finetune_dataset(name='ASL-big', batch_size=80, train=True, samples_pc=20, drop_last=False):
    """
    Creates dataloader for finetuning.
    """

    dataset = FinetuneDataset(f'./datasets/{name}', transform=transform, train=train, samples_pc=samples_pc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=2)

    return dataloader