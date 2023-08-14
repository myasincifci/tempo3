import time

import mlflow
import torch
from lightly.loss import BarlowTwinsLoss
from tqdm import tqdm

from tempo3.data.datasets import (finetune_dataset, video_dataset,
                                  video_dataset_h5)
from tempo3.data.pdfs import pdf_index
from tempo3.models import Tempo
from train_utils import train

DIST = "normal"
PROX = 15
LR = 1e-3
LAMBDA = 1e-2

def main():
    mlflow.log_params({
        "DIST": DIST,
        "PROX": PROX,
        "LR": LR,
        "LAMBDA": LAMBDA
    })

    pdf_f = pdf_index["normal"]

    train_loader = video_dataset_h5(path='./datasets/ASL-big/frames.hdf5', proximity=PROX, batch_size=256, pdf=pdf_f)
    test_loader  = video_dataset_h5(path='./datasets/ASL-big/frames_test.hdf5' , proximity=PROX, batch_size=256, pdf=pdf_f)

    train_loader_ft = finetune_dataset(name='ASL-big', train=True, batch_size=10)
    test_loader_ft = finetune_dataset(train=False, batch_size=10)

    # Use GPU if availabel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    model = torch.compile(Tempo(pretrain=True).to(device))

    start = time.time()

    train(
        model=model,
        epochs=300,
        lr=LR,
        l=LAMBDA,
        train_loader=train_loader,
        test_loader=test_loader,
        train_loader_ft=train_loader_ft,
        test_loader_ft=test_loader_ft,
        device=device
    )

    end = time.time()

    print(f"Training time: {end-start}s")

if __name__ == "__main__":
    main()