import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from hippocampus_segmentation.models.unet import UNet
from hippocampus_segmentation.data.dataset import HippocampusDataset
from hippocampus_segmentation.losses import DiceLoss
from hippocampus_segmentation import config

def train():
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    dataset = HippocampusDataset(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, transform=transform
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = UNet().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = DiceLoss()

    model.train()
    for epoch in range(config.NUM_EPOCHS):
        for images, masks in loader:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            preds = model(images)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {loss.item():.4f}")

    print("Training complete.")
