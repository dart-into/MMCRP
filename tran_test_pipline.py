import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import random
import os
from dataload import CustomImageDataset
from nets.models import BwCombinedInteractModel50
from sklearn.model_selection import StratifiedShuffleSplit

num_classes = 4
batch_size = 32
max_epochs = 100
patience = 20
repeats = 3

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_full = CustomImageDataset("Dataset\chenpi_iphone/full", transform=transform)
dataset_black = CustomImageDataset("Dataset\chenpi_iphone/black", transform=transform)
dataset_white = CustomImageDataset("Dataset\chenpi_iphone/white", transform=transform)
labels = [dataset_full.preprocessed_data[path]['label'] for path in dataset_full.samples]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_once(seed, dataset_full, dataset_black, dataset_white, labels):
    seed_everything(seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    for train_idx, test_val_idx in sss.split(np.zeros(len(labels)), labels):
        temp_labels = [labels[i] for i in test_val_idx]
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        for val_idx_rel, test_idx_rel in sss_val.split(np.zeros(len(temp_labels)), temp_labels):
            val_idx = [test_val_idx[i] for i in val_idx_rel]
            test_idx = [test_val_idx[i] for i in test_idx_rel]

    train_full = Subset(dataset_full, train_idx)
    val_full = Subset(dataset_full, val_idx)
    test_full = Subset(dataset_full, test_idx)

    train_black = Subset(dataset_black, train_idx)
    val_black = Subset(dataset_black, val_idx)
    test_black = Subset(dataset_black, test_idx)

    train_white = Subset(dataset_white, train_idx)
    val_white = Subset(dataset_white, val_idx)
    test_white = Subset(dataset_white, test_idx)

    train_loader = DataLoader(list(zip(train_full, train_black, train_white)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(val_full, val_black, val_white)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(list(zip(test_full, test_black, test_white)), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BwCombinedInteractModel50(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


    best_acc = 0
    counter = 0
    best_model_path = f"best_model_seed_{seed}.pt"

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            (full_img, label, _), (black_img, _, _), (white_img, _, _) = batch
            optimizer.zero_grad()
            outputs = model(full_img.to(device), black_img.to(device), white_img.to(device))
            loss = criterion(outputs, label.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                (full_img, label, _), (black_img, _, _), (white_img, _, _) = batch
                outputs = model(full_img.to(device), black_img.to(device), white_img.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label.to(device)).sum().item()
        acc = correct / total
        print(f"Seed {seed} Epoch {epoch+1}: Val Acc = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} with best val acc {best_acc:.4f}")
            break

    print(f"Loading best model from {best_model_path} for testing")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            (full_img, label, _), (black_img, _, _), (white_img, _, _) = batch
            outputs = model(full_img.to(device), black_img.to(device), white_img.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label.to(device)).sum().item()
    test_acc = correct / total
    print(f"Seed {seed}: Test Acc = {test_acc:.4f}\n")

if __name__ == "__main__":
    for repeat in range(repeats):
        seed = 42 + repeat * 100
        train_once(seed, dataset_full, dataset_black, dataset_white, labels)
