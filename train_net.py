import time

import torch
from lightly.loss import BarlowTwinsLoss
from tqdm import tqdm

from tempo3.data.datasets import video_dataset, video_dataset_h5
from tempo3.data.pdfs import pdf_index
from tempo3.models import Tempo
from train_utils import train

def main():
    pdf_f = pdf_index["uniform"]
    train_loader = video_dataset_h5(path='./datasets/ASL-big/frames_train.hdf5', proximity=15, batch_size=256, pdf=pdf_f)
    test_loader  = video_dataset_h5(path='./datasets/ASL-big/frames_test.hdf5' , proximity=15, batch_size=256, pdf=pdf_f)

    # Use GPU if availabel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    model = torch.compile(Tempo(pretrain=True).to(device))

    start = time.time()

    train(
        model=model,
        epochs=100,
        lr=2e-4,
        l=1e-3,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    end = time.time()

    print(f"Training time: {end-start}s")

if __name__ == "__main__":
    main()