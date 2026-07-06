
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ae.vae import IMAGE_SIZE, celeb_transform, CELEB_PATH, VAE
from dataset_loader import load_celeba
MODEL_FILE = 'ae/vae_torch_celeba/vae_model_20.pth'
class LooseCelebA(CelebA): # Force check integrity to true because pytorch keeps saying it's false
    def _check_integrity(self) -> bool:
        return True
dataset = LooseCelebA("./data", transform=celeb_transform, download=False, split='all')
loader = load_celeba(num_samples=2, image_size=150, normalize=False)
vae:VAE = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)

pics = loader.cpu()


synth, a, b = vae(pics)

pic = synth[0].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
pics = torch.cat((pics, pic), dim=0)

save_image(pics, 'rndpics.jpg', nrow=8)