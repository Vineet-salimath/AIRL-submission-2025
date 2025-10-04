# TiF_fixed.py
# ==============================================================================
# Single-file ViT training script fixed for Windows multiprocessing (DataLoader)
# ==============================================================================

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing

# ----------------------------
# CONFIG
# ----------------------------
config = {
    "dataset": "CIFAR-10",
    "image_size": 32,
    "patch_size": 4,
    "num_classes": 10,
    "embed_dim": 512,
    "depth": 6,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "dropout": 0.1,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 0.05,
    "epochs": 100,
    "warmup_epochs": 10,
    "scheduler": "cosine",
    # set num_workers to 0 if you still hit issues on Windows
    "num_workers": 2,
}

# ----------------------------
# MODEL COMPONENTS
# ----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, depth, num_heads, mlp_ratio, dropout):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_output = x[:, 0]
        output = self.head(cls_token_output)
        return output

# ----------------------------
# TRAIN/TEST FUNCTIONS (no side-effect process spawning)
# ----------------------------
def train_one_epoch(model, device, trainloader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar.set_postfix({
            'loss': f'{train_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': optimizer.param_groups[0]['lr']
        })

def test_one_epoch(model, device, testloader, criterion, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(testloader, desc=f"Epoch {epoch+1} [Test]", leave=False)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar.set_postfix({
                'loss': f'{test_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    return 100. * correct / total

# ----------------------------
# MAIN: create dataloaders, model, optimizer and run training
# ----------------------------
def main():
    print("Preparing data loaders...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(config['image_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # If you still get Windows multiprocessing issues, set num_workers=0 here
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisionTransformer(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=config['dropout']
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-5)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(config['epochs']):
        if epoch < config['warmup_epochs']:
            # simple linear warmup
            lr = config['learning_rate'] * (epoch + 1) / config['warmup_epochs']
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)
        current_acc = test_one_epoch(model, device, testloader, criterion, epoch)

        if current_acc > best_acc:
            best_acc = current_acc
            print(f"New best accuracy: {best_acc:.2f}%. You can save model here if desired.")

        if epoch >= config['warmup_epochs']:
            if config['scheduler'] == 'cosine':
                scheduler.step()

        # optional: print status each epoch
        print(f"Epoch {epoch+1}/{config['epochs']} finished. Current acc: {current_acc:.2f}%. Best: {best_acc:.2f}%")

    end_time = time.time()
    print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.")
    print(f"Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    # On Windows, this helps multiprocessing/DataLoader spawn correctly
    multiprocessing.freeze_support()
    main()
