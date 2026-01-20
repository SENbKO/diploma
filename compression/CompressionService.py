import numpy as np
import cv2
from scipy.fftpack import dct, idct

class DCTService:

    def __init__(self, keep_ratio: float = 0.25):
        self.keep_ratio = keep_ratio

    #Two-dimensional Discrete Cosine Transform
    @staticmethod
    def dct2(block: np.ndarray) -> np.ndarray:
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    #Two-dimensional Inverse dicrete cosine transform
    @staticmethod
    def idct2(block: np.ndarray) -> np.ndarray:
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    #Perform quantization for one channel
    def process_channel(self, channel: np.ndarray) -> np.ndarray:
        dct_channel = self.dct2(channel)

        h, w = dct_channel.shape
        mask = np.zeros_like(dct_channel)
        mask[:int(h * self.keep_ratio), :int(w * self.keep_ratio)] = 1
        print(self.keep_ratio)
        dct_low = dct_channel * mask

        rec_channel = self.idct2(dct_low)
        rec_channel = np.clip(rec_channel, 0, 255)
        return rec_channel

    #Perform quantization for the whole image
    def process_image(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(img.shape) == 2:
            return self.process_channel(img).astype(np.uint8)

        channels = [self.process_channel(img[:, :, i]) for i in range(3)]
        return np.stack(channels, axis=2).astype(np.uint8)
