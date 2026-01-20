import segmentation_models_pytorch as smp
from torch.utils.data import  DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations
import ssl
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryJaccardIndex
from PascalVOC_Dataset import PascalVOC_Dataset
import os
from pathlib import Path
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
ssl._create_default_https_context = ssl._create_unverified_context

def main():

    dataset_path = Path(__file__).parent / ".." / "dataset" / "VOC2012_train_val" / "VOC2012_train_val"
    text_path = dataset_path / "ImageSets" / "Segmentation" / "trainval.txt"
    with open(text_path, "r") as f:
        image_ids = f.read().splitlines()
        print(image_ids)

    image_directory= dataset_path / "JPEGImages"
    mask_directory = dataset_path / "SegmentationClass"

    image_paths = [image_directory / f"{img_id}.jpg" for img_id in image_ids]
    mask_paths = [mask_directory/f"{img_id}.png" for img_id in image_ids]

    print(f"Images length: {len(image_paths)}")
    print(f"Masks length: {len(mask_paths)}")

    #Augmentation for images
    trasform = albumentations.Compose([
        albumentations.Resize(256,256),
        albumentations.Rotate(p=1.0, limit=35),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.1),
        albumentations.Normalize(),
        ToTensorV2()
    ])

    train_dataset = PascalVOC_Dataset(image_paths, mask_paths, transform=trasform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Backbone initialization
    unet = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    ).to(device)



    #Training parameters
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    iou_metric = BinaryJaccardIndex()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3 )
    NUM_EPOCHS = 50
    for epoch in range(NUM_EPOCHS):
        unet.train()
        epoch_loss = 0.0
        total_iou = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = unet(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).int()
            targets = masks.int()

            iou = iou_metric(preds, targets)
            epoch_loss += loss.item()
            total_iou += iou.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_iou = total_iou/len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, IoU: {avg_iou}")
        scheduler.step(avg_iou)
    #Save the model    
    torch.save(unet.state_dict(), 'unet_impl_iou_50_epochs.pth')

if __name__ == '__main__':
    main()