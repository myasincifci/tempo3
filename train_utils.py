import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from lightly.loss import BarlowTwinsLoss, NTXentLoss, SwaVLoss,SymNegCosineSimilarityLoss, DINOLoss
from mlflow import log_metric
from tqdm import tqdm

from sklearn.svm import SVC


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
def test_model(model, test_reps, test_dataset, device):

    model.eval()

    wrongly_classified = 0
    for repr, label in test_reps:
        total = repr.shape[0]

        inputs, labels = repr.to(device), label.to(device)

        with torch.no_grad():
            preds = model(inputs).argmax(dim=1)

        wrong = (total - (preds == labels).sum()).item()
        wrongly_classified += wrong

    model.train()

    return 1.0 - (wrongly_classified / len(test_dataset))

def linear_svm(model: torch.nn.Module, train_loader, test_loader, device):
    model.eval()

    X_train, T_train = [], []
    X_test , T_test  = [], []
    with torch.no_grad():
        for input, label in train_loader:
            repr = model.backbone(input.to(device)).detach()
            repr = torch.flatten(repr, start_dim=1)
            
            X_train.append(repr.cpu())
            T_train.append(label.cpu())

        for input, label in test_loader:
            repr = model.backbone(input.to(device)).detach()
            repr = torch.flatten(repr, start_dim=1)
            
            X_test.append(repr.cpu())
            T_test.append(label.cpu())

    X_train = torch.cat(X_train, dim=0).detach().numpy()
    T_train = torch.cat(T_train, dim=0).detach().numpy()

    X_test = torch.cat(X_test, dim=0).detach().numpy()
    T_test = torch.cat(T_test, dim=0).detach().numpy()

    clf = SVC(gamma="auto")
    clf.fit(X_train, T_train)
    
    Y_test = clf.predict(X_test)

    return (Y_test == T_test).sum() / len(T_test)


def linear_eval(model: torch.nn.Module, train_loader, test_loader, device):
    model.eval()

    reps = []
    test_reps = []
    with torch.no_grad():
        for input, label in train_loader:
            repr = model.backbone(input.to(device)).detach()
            repr = torch.flatten(repr, start_dim=1)
            reps.append((repr, label.to(device)))

        for input, label in test_loader:
            repr = model.backbone(input.to(device)).detach()
            repr = torch.flatten(repr, start_dim=1)
            test_reps.append((repr, label.to(device)))

    linear_head = nn.Linear(in_features=512, out_features=24, bias=True).to(
        device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        linear_head.parameters(), lr=0.02, weight_decay=0.0001)

    linear_head.train()
    for epoch in range(100):
        for repr, label in reps:
            labels = nn.functional.one_hot(label, num_classes=24).float()
            inputs, labels = repr.to(device), labels.to(device)

            pred = linear_head(inputs)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    error = test_model(linear_head, test_reps, test_loader.dataset, device)

    return error


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


def train(model: torch.nn.Module, epochs, lr, l, train_loader, test_loader, train_loader_ft, test_loader_ft, device):
    criterion = BarlowTwinsLoss(lambda_param=l)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1, last_epoch=-1)

    for epoch in range(epochs):
        log_metric("lr", optimizer.param_groups[0]["lr"], epoch)

        train_loss = train_one_epoch(model, train_loader,
                                     criterion, optimizer, device)
        log_metric("train_loss", train_loss, epoch)

        # test_loss = eval(model, criterion, test_loader, device)
        # log_metric("test_loss", test_loss, epoch)

        # linear_accuracy = linear_eval(model, train_loader_ft, test_loader_ft, device)
        # log_metric("linear_accuracy", linear_accuracy, epoch)
        
        svm_accuracy = linear_svm(model, train_loader_ft, test_loader_ft, device)
        log_metric("svm_accuracy", svm_accuracy, epoch)
        # scheduler.step()

    return model.backbone.state_dict()
