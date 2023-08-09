import torch
from lightly.loss import BarlowTwinsLoss
from mlflow import log_metric
from tqdm import tqdm


@torch.no_grad()
def eval(model: torch.nn.Module, criterion, test_loader, device):
    model.eval()

    losses = []
    for image, image_d, _, _ in tqdm(test_loader):
        image = image.to(device)
        image_d = image_d.to(device)

        z0 = model(image)
        z1 = model(image_d)
        loss = criterion(z0, z1)
        losses.append(loss.item())

    avg_loss = sum(losses)/len(losses)

    return avg_loss


@torch.no_grad()
def linear_eval():
    pass

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

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


def train(model: torch.nn.Module, epochs, lr, l, train_loader, test_loader, device):
    criterion = BarlowTwinsLoss(lambda_param=l)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader,
                                     criterion, optimizer, device)
        log_metric("train_loss", train_loss, epoch)

        test_loss = eval(model, criterion, test_loader, device)
        log_metric("test_loss", test_loss, epoch)


    return model.backbone.state_dict()
