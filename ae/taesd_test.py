import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt 
import numpy as np

# Based on : ae/taesd/examples/Encoding_and_Decoding.ipynb

import torch

from PIL import Image
import torchvision.transforms.functional as TF

from ae.taesd.taesd import TAESD


dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device", dev)

taesd_params = [
    ["ae/taesd/taesd_encoder.pth","ae/taesd/taesd_decoder.pth"],    #0
    ["ae/taesd/taef1_encoder.pth","ae/taesd/taef1_decoder.pth"],    #1
    ["ae/taesd/taef2_encoder.pth","ae/taesd/taef2_decoder.pth"],    #2
    ["ae/taesd/taesana_encoder.pth","ae/taesd/taesana_decoder.pth"],#3
    ["ae/taesd/taesd3_encoder.pth","ae/taesd/taesd3_decoder.pth"],  #4 
    ["ae/taesd/taesdxl_encoder.pth", "ae/taesd/taesdxl_decoder.pth"]#5
    ]
i = 0
taesd = TAESD(*taesd_params[i]).to(dev)

image_path = r"C:\Users\Mahe\Development\GitHub\RMGD\results\algo_9_conv_mosaic_36369022.png"
test_image = TF.center_crop(TF.resize(Image.open(image_path).convert("RGB"), 512), 512)

def display(img, idx):
    plt.figure(idx)
    plt.imshow(img)

# plt.imshow(test_image)
# plt.show()

def summarize_tensor(x):
    return f"\033[34m{str(tuple(x.shape)).ljust(24)}\033[0m (\033[31mmin {x.min().item():+.4f}\033[0m / \033[32mmean {x.mean().item():+.4f}\033[0m / \033[33mmax {x.max().item():+.4f}\033[0m)"

def latent_to_visualization(latent):
    latent = TAESD.scale_latents(latent)
    return torch.cat([latent[:3], latent[3:].expand(3, *latent.shape[-2:])], -2)

def encode_dataset(image):
    image_raw = TF.to_tensor(image).unsqueeze(0).to(dev)
    image_enc = taesd.encoder(image_raw)

def demo_taesd_on_image_og(taesd, image, dev):
    image_raw = TF.to_tensor(image).unsqueeze(0).to(dev)
    image_enc = taesd.encoder(image_raw)
    image_dec = taesd.decoder(image_enc).clamp(0, 1)
    
    image_diff = torch.norm((image_raw - image_dec), p=2 ,dim=1)
    
    print("input image", summarize_tensor(image_raw[0]))
    display(TF.to_pil_image(image_raw[0]),0)

    print("latents", summarize_tensor(image_enc[0]))
    print("(these latents are the same size / scale as SD UNet-generated latents - no extra scale_factor is needed)")
    display(TF.to_pil_image(latent_to_visualization(image_enc[0])),1)
    
    print("decoded image", summarize_tensor(image_dec[0]))
    display(TF.to_pil_image(image_dec[0]),2)
    print(image_diff.shape)
    print("difference image", summarize_tensor(image_diff[0]))
    display(TF.to_pil_image(image_diff[0]),3)
    
def demo_taesd_on_image(taesd, image, dev):
    image_raw = TF.to_tensor(image).unsqueeze(0).to(dev)
    image_enc = taesd.encoder(image_raw)
    image_dec = taesd.decoder(image_enc).clamp(0, 1)
    
    image_diff = torch.norm((image_raw - image_dec), p=2 ,dim=1)
    
    print("input image", summarize_tensor(image_raw[0]))

    print("latents", summarize_tensor(image_enc[0]))
    
    print("decoded image", summarize_tensor(image_dec[0]))

    print("difference image", summarize_tensor(image_diff[0]))

    fig,axes = plt.subplots(2, 2, figsize=(2, 2), layout="constrained")
    axs_flat = np.ndarray.flatten(axes)
    axs_flat[0].imshow(TF.to_pil_image(image_raw[0])) 
    axs_flat[0].set_title("Input Image (II)")
    
    try :
        axs_flat[1].imshow(TF.to_pil_image(latent_to_visualization(image_enc[0])))
        axs_flat[1].set_title("Latent visualizatrion")
    except : 
        print("Can't display latent view")
    axs_flat[2].imshow(TF.to_pil_image(image_dec[0]))
    axs_flat[2].set_title("Decoded Image (DI)")
    
    axs_flat[3].imshow(TF.to_pil_image(image_diff[0]))
    axs_flat[3].set_title("II / DI Difference")
    
    [ax.axis('off') for ax in axs_flat]
demo_taesd_on_image(taesd, test_image, dev)

plt.show()