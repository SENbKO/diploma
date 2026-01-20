import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp


class BackgroundBlurModel:    
    def __init__(
        self,
        weights_path: str,
        device: str | None = None,
        input_size: tuple[int, int] = (256, 256),
        threshold: float = 0.5,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_size = input_size
        self.threshold = threshold

        # Load model
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.eval()

    #Get the segmentation mask and apply blur
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: RGB image as NumPy array (H, W, 3)
        Returns:
            RGB image with blurred background
        """
        mask = self._predict_mask(image)
        result = self._apply_blur(image, mask)
        return result

    
    def _predict_mask(self, image: np.ndarray) -> np.ndarray:
        original_h, original_w = image.shape[:2]

        # Preprocess
        resized = cv2.resize(image, self.input_size)
        tensor = (
            torch.from_numpy(resized / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            output = self.model(tensor)
            output = torch.sigmoid(output)
            mask = (output > self.threshold).float()

        mask = mask.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (original_w, original_h))

        return mask

    def _apply_blur(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_3ch = np.stack([mask] * 3, axis=-1)
        blurred = self._gaussian_blur_per_channel(image, sigma=5, kernel_size=17)
        result = np.where(mask_3ch > 0.5, image, blurred)
        return result.astype(np.uint8)

    def _gaussian_blur_per_channel(
        self, image: np.ndarray, sigma: float, kernel_size: int
    ) -> np.ndarray:
        channels = [
            self._gaussian_filter(image[:, :, c], kernel_size, sigma)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2)


    def _define_gaussian_kernel(self, kernel_size: int, sigma: float) -> np.ndarray:
        one_dim = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(one_dim, one_dim)
        #Formula of Gaussian distribution
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    def _gaussian_filter(
        self, image: np.ndarray, kernel_size: int, sigma: float
    ) -> np.ndarray:
        padding = kernel_size // 2
        padded = np.pad(image, ((padding, padding), (padding, padding)), mode="reflect")
        new_image = np.zeros_like(image)
        kernel = self._define_gaussian_kernel(kernel_size, sigma)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i : i + kernel_size, j : j + kernel_size]
                new_image[i, j] = np.sum(window * kernel)

        return new_image
