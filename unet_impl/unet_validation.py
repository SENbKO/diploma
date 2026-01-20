import segmentation_models_pytorch as smp
from torch.utils.data import  DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations
import ssl
import torch
from unet_impl.PascalVOC_Dataset import PascalVOC_Dataset
from torchmetrics.classification import BinaryJaccardIndex
from pathlib import  Path

ssl._create_default_https_context = ssl._create_unverified_context
def main():
    dataset_path = Path(__file__).parent / ".." / "dataset" / "VOC2012_train_val" / "VOC2012_train_val"
    text_path = dataset_path / "ImageSets" / "Segmentation" / "val.txt"
    with open(text_path, "r") as f:
        image_ids = f.read().splitlines()

    image_directory = dataset_path / "JPEGImages"
    mask_directory = dataset_path / "SegmentationClass"

    image_paths = [image_directory/ f"{img_id}.jpg" for img_id in image_ids]
    mask_paths = [mask_directory/f"{img_id}.png" for img_id in image_ids]

    print(f"Images length: {len(image_paths)}")
    print(f"Masks length: {len(mask_paths)}")

    trasform = albumentations.Compose([
        albumentations.Resize(256,256),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Normalize(),
        ToTensorV2()
    ])


    val_dataset = PascalVOC_Dataset(image_paths, mask_paths, transform=trasform)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    ).to(device)

    unet_weights  = Path(__file__).parent  / 'unet_impl_iou_50_epochs.pth'
    unet.load_state_dict(
        torch.load(unet_weights, map_location=device)
    )
    unet.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    iou_metric = BinaryJaccardIndex()

    val_loss = 0.0
    iou_scores = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = unet(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).int()
            iou = iou_metric(preds, masks.int())
            iou_scores.append(iou.item())

    avg_loss = val_loss / len(val_loader)
    avg_iou = sum(iou_scores) / len(iou_scores)

    print(f"Validation Loss: {avg_loss:.4f} | Accuracy: {avg_iou:.4f}")



if __name__ == '__main__':
    main()