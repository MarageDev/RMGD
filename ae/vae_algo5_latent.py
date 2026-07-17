import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

PATCHSIZE = 8
STRIDE = 4 #todo

@torch.no_grad()
def extract_centered_patches(img, patchsize):
    """
    Extrait les patches centrés pour chaque pixel de l'image.
    """
    pad = patchsize // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    return F.unfold(img_padded, kernel_size=patchsize, padding=0, stride=STRIDE)

@torch.no_grad()
def weighted_patch_average(P_synth, patchsize,C, H, W, mode="standard", device='cpu'):
    
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

    fold_layer = nn.Fold((W, H), kernel_size=patchsize, dilation=1, padding=patchsize//2, stride=STRIDE)
    
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
def algo5(D_train, patchsize=3, N=50, schedule='linear', device='cpu', mask_weight_type="standard",seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    D_train = D_train.to(device)
    N_imgs, C, H, W = D_train.shape
    
    # Extract patches from training data
    Z = extract_centered_patches(D_train, patchsize)  # (N_imgs, C*patchsize^2, H*W)
    
    # Initialize with Gaussian noise
    x_noise = torch.randn(1, C, H, W, device=device)
    x_n1 = x_noise.clone()
    
    # Save initial noise
    saved_steps = [x_n1.cpu()]
    
    times = make_times(N, schedule, 0)
    
    for it in tqdm(range(N)):
        t = times[it]

        delta_t = times[it + 1] - t 
        

        x_patches = extract_centered_patches(x_n1, patchsize)  # (1, C*patchsize^2, H*W)
        

        dists = torch.sum((x_patches - Z*t)**2, dim=1)  # (N_imgs, H*W)

        w = torch.softmax(-dists / (2 * ((1 - t) ** 2)), dim=0)  # (N_imgs, H*W)
        
        v = ((Z - x_patches) * w.unsqueeze(1)).sum(0, keepdim=True) / (1 - t)
        # Z : (N_imgs, C*patchsize^2, H*W)  // x_patches (1, C*patchsize^2, H*W) // w (N_imgs, H*W)
        
        
        x_patches_updated = x_patches + v * delta_t

        x_n1 = weighted_patch_average(x_patches_updated, patchsize, C,H, W,mode=mask_weight_type ,device=device)
        
        saved_steps.append(x_n1.cpu())
    
    return saved_steps

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ae.taesd.taesd import TAESD
def transform_to_latent_space(tensor, taesd:TAESD, dev):
    """
    Output : N samples, 4, H/(2^upscaling number), W/(2^upscaling number)
    """
    
    from PIL import Image
    import torchvision.transforms.functional as TF

    encoded_tensor = taesd.encoder(tensor[0].unsqueeze(0).to(dev))
    cat_tensor = torch.zeros(tensor.shape[0],*encoded_tensor.shape[1:]).to(dev)
    cat_tensor[0] = encoded_tensor
    for i in tqdm(range(1,tensor.shape[0]),"Encoding tensors"):
        encoded_tensor = taesd.encoder(tensor[i].unsqueeze(0).to(dev))
        cat_tensor[i] = encoded_tensor
    return cat_tensor


def imgs_to_gif(path="out.gif",imgs=None):
    from PIL import Image
    
    np_imgs = [np.uint8(np.clip(img.permute(0, 2, 3, 1).detach().numpy()[0] , 0, 1) * 255) for img in imgs]
    im_list = [Image.fromarray(np_img, mode='RGB') for np_img in np_imgs]
    im_list[0].save(path, save_all=True, append_images=im_list[1:], duration=1, loop=0)


def imgs_to_gif_encode(imgs=None):
    from PIL import Image
    
    imgs_encode_to_rgb = [ TAESD.scale_latents(i.to(device))[0][:3].permute(1,2,0).detach().cpu().numpy() for i in imgs]

    np_imgs = [np.uint8(img * 255) for img in imgs_encode_to_rgb]
    im_list = [Image.fromarray(np_img, mode='RGB') for np_img in np_imgs]
    im_list[0].save("out_a5_enc.gif", save_all=True, append_images=im_list[1:], duration=1, loop=0)

def decode_tensor(tensor:torch.TensorType, taesd:TAESD):
    return taesd.decoder(tensor).clamp(0, 1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from data_tensor_loader import load_data_to_tensor
    from algorithms.novelty import *
    torch.cuda.empty_cache()
    device = "cpu" if False else "cuda:0"
    dataset_loading_parameters = {
        "data_set_name" : "c",
        "num_samples" : 40,
        "target_labels" : [4],
        "image_size" : 256, 
        "normalize" : False,  
    }
    data_tensor = load_data_to_tensor(None, data_dir="./data", dataset_loading_parameters=dataset_loading_parameters, force_reload_tensor=False)
    #tensor_file = "tensor_cache_0_9905840eee87d2da9e032f2c96eee9ce0a1743a4acad78f534f992ebd0eadbf7.pt"
    #tensor_file = "tensor_cache_0_2686aeef7861d14eedc6f027f50b416ed0fc1d107f71c309c4a7d7ce3a0af441.pt"
    #tensor_path = f"./data/saved_tensors/{tensor_file}"
    #data_tensor = torch.load(tensor_path, map_location=device)
    
    taesd = TAESD(*["ae/taesd/taesd_encoder.pth","ae/taesd/taesd_decoder.pth"]).to(device)
    encoded_tensor = transform_to_latent_space(data_tensor,taesd=taesd, dev=device)
    a5 = algo5(encoded_tensor, patchsize=PATCHSIZE, N=25, device=device, mask_weight_type="",schedule='linear', seed=None)
    image_dec = decode_tensor(a5[-1].to(device), taesd).cpu()
    try : 
        for i, t in enumerate(encoded_tensor[:5]):
            plt.figure(i)


            latent = TAESD.scale_latents(t) # 4 32 32   
            #plt.imshow(TF.to_pil_image(torch.cat([latent[:3], latent[3:].expand(3, *latent.shape[-2:])], -2)[0]))
            #out = TF.to_pil_image(latent[:3])
            out = latent[:3].permute(1,2,0).detach().cpu().numpy()
            plt.imshow(out)
    except :
        pass
    
    #plt.imsave(f"nifty_alg_5{1}.png",np.clip((a5[0].permute(0, 2, 3, 1).numpy()[0] + 1) / 2, 0, 1))
    #plt.imsave(f"nifty_alg_5{2}.png",np.clip((a5[-1].permute(0, 2, 3, 1).numpy()[0] + 1) / 2, 0, 1))
    
    imgs_to_gif("out_a5.gif",[decode_tensor(i.to(device), taesd).cpu() for i in a5])
    imgs_to_gif_encode(a5)
    
    
    
    
    plt.figure(4)
    plt.tight_layout()
    plt.imshow(TF.to_pil_image(image_dec[0].clamp(0,1)))
    
    comparison = compare_ref_stack(
        image_dec.squeeze(0).to(device), 
        data_tensor.to(device),
        threshold=0.5, 
        distance_gradient=True,
        spatial_weights=None,
        smooth_kernel=3
    )
    
    plt.figure(5)
    plt.title("Patch Regions")
    plt.imshow(tensor_to_numpy_img(comparison[0]))
    
    plt.figure(6)
    plt.title("Mosaic View of Patches")
    plt.imshow(tensor_to_numpy_img((comparison[1]+1)/2))
    
    plt.show()
    
