import torch
import matplotlib.pyplot as plt
import cv2
import segmentation_models_pytorch as smp
import numpy as np


def main():
    # Load and preprocess input image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = cv2.imread("2008_000030.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (256, 256))
    image_tensor = torch.tensor(image_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)


    unet = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        ).to(device)
    unet.load_state_dict(torch.load('unet_impl_iou_50_epochs.pth'))
    # Run inference
    unet.eval()
    with torch.no_grad():
        output = unet(image_tensor)
        output = torch.sigmoid(output)
        mask_pred = (output > 0.5).float()

    # Plot image and predicted mask
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    print(mask_pred.squeeze().cpu().numpy())
    original_size = (image.shape[1], image.shape[0])
    mask = cv2.resize(mask_pred.squeeze().cpu().numpy(), original_size)
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    print(image.shape)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    blurred = apply_gaussian_blue_per_channel(image, sigma=5, kernel_size=17)
    result = np.where(mask_3ch, image, blurred)
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title("Blurred image")
    plt.axis("off")

    plt.show()

def apply_gaussian_blue_per_channel(image,sigma : float=1.0, kernel_size : int = 5):
    channels = [apply_gaussian_filter(image[:, :, c], kernel_size, sigma) for c in range(image.shape[2])]
    return np.stack(channels, axis=2)

def define_gaussian_kernel(kernel_size : int, sigma : float):
    one_dim = np.linspace(-(kernel_size//2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(one_dim, one_dim)
    kernel = np.exp(-(xx**2 + yy**2) / (2 *sigma**2))
    kernel = kernel/np.sum(kernel)
    return kernel

def apply_gaussian_filter(image: np.ndarray, kernel_size, sigma):
    padding_size = kernel_size // 2
    padded_image = np.pad(image, ((padding_size, padding_size), (padding_size, padding_size)), mode='reflect')
    new_image = np.zeros_like(image)
    kernel = define_gaussian_kernel(kernel_size, sigma)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+kernel_size, j:j+kernel_size]
            blurred_pixel = np.sum(window*kernel)
            new_image[i, j] = blurred_pixel
    return new_image


if __name__ == '__main__':
    main()
