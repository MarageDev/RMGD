# Based on algorithm 5 but in latent space of TAESD

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


from algorithms.taesd.taesd import TAESD

PATCHSIZE = 12
STRIDE = 6
ITERATIONS = 50

INITIAL_NOISE_FACTOR = 0.5

INIT_PC = 1
INIT_PC_FACTOR = 0.5

ALGO_SAMPLES = 500

SEED = None


OUTPUT_RGB_GIF = True
OUTPUT_LATENT_GIF = False

@torch.no_grad()
def extract_centered_patches(img, patchsize, stride:int = 1):
    """
    Extrait les patches centrés pour chaque pixel de l'image.
    """
    img_padded = img
    return F.unfold(img_padded, kernel_size=patchsize, padding=0, stride=stride)

@torch.no_grad()
def weighted_patch_average(P_synth, patchsize,C, H, W, mode="standard", device='cpu', stride: int = 1):
    
    if mode == "gaussian":
        # Gaussian weight
        half_win_size = patchsize // 2
        sig_pix = 1. * half_win_size
        
        # arange creates the spatial spread
        dx = torch.arange(-half_win_size, half_win_size + 1, 1.0, device=device)
        w1d = torch.exp(-(dx / sig_pix)**2 / 2.0)
        
        # Create 2D weight and adapt to 3 color channels
        w2d = w1d.view(-1, 1) * w1d.view(1, -1)
        w = w2d.repeat(C, 1, 1).view(-1) # 
        w = w.unsqueeze(0).unsqueeze(-1) # (1, C * patchsize^2, 1)
        
    else: # standard
        # NIFTY weight
        w=torch.exp(-torch.linspace(-patchsize//2,patchsize//2,steps=patchsize).pow(2)/2/(patchsize*1/4)**2).to(device)
        w = w.view(-1, 1) * w.view(1, -1)
        w = w.repeat(C, 1, 1).view(-1) # 
        w /= w.sum() # Normalization
        w = w.unsqueeze(0).unsqueeze(-1)

    fold_layer = nn.Fold((W, H), kernel_size=patchsize, dilation=1, padding=0, stride=stride)

    # Apply weights and fold
    synth = fold_layer(P_synth * w)
    count = fold_layer(P_synth * 0 + w)


    count= (count*(count!=0)+1.*(count==0))
    synth = synth / count
    
    return synth

def make_times(n_timestep, schedule='cosine', t0=0): 
    '''
    different time discretizations (0 to 1), 'quad' has smaller timesteps near t=0
    '''
    times = torch.linspace(t0, 1., n_timestep + 1) # default fallback to linear
    if schedule == "linear":
        times = torch.linspace(t0, 1., n_timestep + 1) 
        
    elif schedule == "quad":
        times = torch.linspace(t0 ** 0.5, 1., n_timestep + 1) ** 2

    elif schedule == "cosine":
        times = (
            torch.linspace(torch.arcsin(torch.tensor(t0) ** 0.5), torch.pi / 2, n_timestep + 1) 
        )
        times = torch.sin(times).pow(2)
        times = times / times[-1]
        
    return times

@torch.no_grad()
def ALGO_LATENT(clean_batch_tensor, initialisation_tensor, patchsize=3, N=50, schedule='linear', device='cpu', mask_weight_type="standard", seed=None, stride:int=1):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    clean_batch_tensor = clean_batch_tensor.to(device)
    N_imgs, C, H, W = clean_batch_tensor.shape
    
    # Extract patches from reference dataset
    Z = extract_centered_patches(clean_batch_tensor, patchsize, stride=stride)
    x_noise = initialisation_tensor 

    x_noise = (1. - INITIAL_NOISE_FACTOR ) * x_noise + INITIAL_NOISE_FACTOR * torch.randn_like(x_noise, device=device)
    #t0 = (1. - INITIAL_NOISE_FACTOR )
    t0=0.
    
    x_n1 = x_noise.clone()
    
    saved_steps = [x_n1.cpu()]
    
    times = make_times(N, schedule, t0)
    
    for it in tqdm(range(N)):
        t = times[it]
        delta_t = times[it + 1] - t 

        x_patches = extract_centered_patches(x_n1, patchsize, stride=stride) # (1, C*patchsize^2, H*W) 
        
        dists = torch.sum((x_patches - Z * t)**2, dim=1) # (N_imgs, H*W)

        w = torch.softmax(-dists / (2 * ((1. - t) ** 2)), dim=0)# (N_imgs, H*W)  

        v = ((Z - x_patches) * w.unsqueeze(1)).sum(0, keepdim=True) / (1. - t)
        # Z : (N_imgs, C*patchsize^2, H*W)  // x_patches (1, C*patchsize^2, H*W) // w (N_imgs, H*W)
        
        x_patches_updated = x_patches + v * delta_t
        
        x_n1 = weighted_patch_average(x_patches_updated, patchsize, C, H, W, mode=mask_weight_type, device=device, stride=stride)
        
        yield x_n1.cpu()
        saved_steps.append(x_n1.cpu())
    
    return saved_steps


@torch.no_grad()
def imgs_to_gif(path="out.gif",imgs=None):
    from PIL import Image
    
    np_imgs = [np.uint8(np.clip(img.permute(0, 2, 3, 1).detach().numpy()[0] , 0, 1) * 255) for img in imgs]
    im_list = [Image.fromarray(np_img, mode='RGB') for np_img in np_imgs]
    im_list = im_list + [im_list[-1]] * 10
    im_list[0].save(path, save_all=True, append_images=im_list[1:], duration=5, loop=0)

@torch.no_grad()
def imgs_to_gif_encode(imgs=None, device='cuda:0'):
    from PIL import Image
    
    imgs_encode_to_rgb = [ TAESD.scale_latents(i.to(device))[0][:3].permute(1,2,0).detach().cpu().numpy() for i in imgs]

    np_imgs = [np.uint8(img * 255) for img in imgs_encode_to_rgb]
    im_list = [Image.fromarray(np_img, mode='RGB') for np_img in np_imgs]
    im_list = im_list + [im_list[-1]] * 100
    im_list[0].save("out_a5_enc.gif", save_all=True, append_images=im_list[1:], duration=1, loop=0)

@torch.no_grad()
def transform_to_latent_space(tensors, taesd:TAESD, dev):
    from PIL import Image
    import torchvision.transforms.functional as TF

    # Scale latents immediately after encoding
    encoded_tensor = taesd.encoder(tensors[0].unsqueeze(0).to(dev))
    cat_tensor = torch.zeros(tensors.shape[0], *encoded_tensor.shape[1:]).to(dev)
    cat_tensor[0] = encoded_tensor
    for i in tqdm(range(1, tensors.shape[0]), "Encoding tensors"):
        encoded_tensor = taesd.encoder(tensors[i].unsqueeze(0).to(dev))
        cat_tensor[i] = encoded_tensor
    return cat_tensor

@torch.no_grad()
def transform_to_latent_space_tensor(tensor, taesd:TAESD, dev):
    import torchvision.transforms.functional as TF
    encoded_tensor = taesd.encoder(tensor.to(dev))
    return encoded_tensor

@torch.no_grad()
def decode_tensor(tensor:torch.TensorType, taesd:TAESD):
    return taesd.decoder(tensor).clamp(0, 1)
