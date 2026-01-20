import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class SRDataset(Dataset):
    def __init__(self, hr_dir, scale=2, crop_size=96, augment=True):
        self.hr_dir = hr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.scale = scale
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.hr_images)

    #Perform augmentaton
    def random_augment(self, img):
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.5:
            img = TF.vflip(img)
        if random.random() < 0.5:
            img = img.rotate(90)
        return img

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        #Convert to YUV color pannel
        hr = Image.open(hr_path).convert("YCbCr")

        # Random crop of high resolution image
        x = random.randint(0, hr.width - self.crop_size)
        y = random.randint(0, hr.height - self.crop_size)
        hr = hr.crop((x, y, x + self.crop_size, y + self.crop_size))

        if self.augment:
            hr = self.random_augment(hr)

        # Split channels
        y_hr, cb, cr = hr.split()

        # Downscale Y channel only
        lr_size = self.crop_size // self.scale
        y_lr = y_hr.resize((lr_size, lr_size), Image.BICUBIC)

        y_lr = TF.to_tensor(y_lr)
        y_hr = TF.to_tensor(y_hr)

        return y_lr, y_hr
