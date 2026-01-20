import torch
import torchvision.transforms as T
from PIL import Image
from upscaling_model.FSRCNN import FSRCNN
from torchvision.transforms.functional import to_pil_image


class UpscalingModel:
    def __init__(self, weights_path, scale_factor=2, device="cpu"):
        self.device = torch.device(device)
        self.scale_factor = scale_factor

        self.model = FSRCNN(scale_factor=scale_factor).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

    @torch.no_grad()
    def transform(self, image: Image.Image) -> Image.Image:

        ycbcr = image.convert("YCbCr")
        y, cb, cr = ycbcr.split()

        y_tensor = self.to_tensor(y).unsqueeze(0).to(self.device)

        y_sr = self.model(y_tensor)
        print("upscaling complete 1")
        y_sr = y_sr.squeeze(0).clamp(0, 1)

        _, h, w = y_sr.shape

        cb_up = cb.resize((w, h), Image.BICUBIC)
        cr_up = cr.resize((w, h), Image.BICUBIC)

        y_sr_img = to_pil_image(y_sr.cpu()).convert("L")

        output = Image.merge("YCbCr", (y_sr_img, cb_up, cr_up)).convert("RGB")

        return output
