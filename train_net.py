import torch
from lightly.loss import BarlowTwinsLoss
from tqdm import tqdm

from tempo3.data.datasets import video_dataset, video_dataset_h5
from tempo3.data.pdfs import pdf_index
from tempo3.models import Tempo

import time

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    losses = []
    for image, image_d, _, _ in tqdm(dataloader):
        image = image.to(device)
        image_d = image_d.to(device)

        z0 = model(image)
        z1 = model(image_d)
        loss = criterion(z0, z1)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = sum(losses)/len(losses)

    return avg_loss


def train(model, epochs, lr, l, train_loader, device):
    criterion = BarlowTwinsLoss(lambda_param=l)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)

    for epoch in range(epochs):
        print(optimizer.param_groups[0]['lr'])
        loss = train_one_epoch(model, train_loader,
                               criterion, optimizer, device)
        print(epoch, loss)

    return model.backbone.state_dict()


def main():
    pdf_f = pdf_index["uniform"]
    train_loader = video_dataset_h5(proximity=15, batch_size=200, pdf=pdf_f)

    # Use GPU if availabel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    model = Tempo(pretrain=True).to(device)

    start = time.time()

    train(
        model=model,
        epochs=10,
        lr=1e-3,
        l=1e-3,
        train_loader=train_loader,
        device=device
    )

    end = time.time()

    print(f"Training time: {end-start}s")

if __name__ == "__main__":
    main()