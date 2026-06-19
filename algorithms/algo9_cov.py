# Algo 7 un peu plus optimisé + stride
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def extract_centered_patches(img, patchsize, stride=1):
    """
    Extrait les patches centrés pour chaque pixel de l'image + stride.
    """
    pad = patchsize // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    return F.unfold(img_padded, kernel_size=patchsize, padding=0, stride=stride)

@torch.no_grad()
def get_spatial_weights(patchsize, C, mode="standard", device='cpu'):
    """
    Précalcule des poids spatiaux (Gaussien ou Standard)
    """
    if mode == "gaussian":
        # Gaussian weight
        half_win_size = patchsize // 2
        sig_pix = 0.5 * half_win_size
        
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

    return w, w2d #w2d used for comparison when the 2d weight distribution is needed 

@torch.no_grad()
def weighted_patch_average(P_synth, patchsize, C, H, W, spatial_weights, device='cpu', stride=1):
    # d'après la doc, apparament le fold prend du (H, W) et pas du (W, H)
    fold_layer = nn.Fold((H, W), kernel_size=patchsize, dilation=1, padding=patchsize//2, stride=stride)

    synth = fold_layer(P_synth * spatial_weights)

    count_ones = torch.ones_like(P_synth)
    count = fold_layer(count_ones * spatial_weights)

    count = torch.where(count == 0, torch.tensor(1.0, device=device), count) # avec torch.where, ça doit être plus rapide
    synth = synth / count
    
    return synth

def make_times(n_timestep, schedule='cosine', t0=0): 
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

def calculate_total_steps(N, upsamples, renoise_factor):
    total_steps = 1
    for s in range(upsamples):
        if s == 0:
            total_steps += N
        else:
            n_steps_upsampled = int(N * renoise_factor)
            total_steps += n_steps_upsampled
    return total_steps

@torch.no_grad()
def algo8(D_train, patchsize=3, stride=1, N=50, schedule='linear', device='cpu', mask_weight_type="standard", renoise_factor=0.1, save_immediatly_to_cpu=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    N_imgs, C, H, W = D_train[0].shape # Initial shape of the refence image dataset
    scales = len(D_train)
    
    # Initialize with Gaussian noise
    x_noise = torch.randn(1, C, H, W, device=device)
    x_n1 = x_noise.clone()
    
    # Allouer la mémoire de la liste au début
    total_steps_count = calculate_total_steps(N, scales, renoise_factor)
    saved_steps = [None for _ in range(total_steps_count)] 
    
    
    # Précalculer les poids spatiaux
    spatial_weights = get_spatial_weights(patchsize, C, mode=mask_weight_type, device=device)[0]
    
    t0  = 0
    step = 0
    for s in range(scales):
        D_train_resized = D_train[s].to(device)
        H_resized, W_resized =  H * (2**(s)), W * (2**(s))
        
        
        
        
        

        if s == 0: 
            t0 = 0

            # Mat cov calc
            mu = D_train_resized.mean(dim=(0), keepdim=True) # (1, C, H, W)
            mu_patches = extract_centered_patches(mu, patchsize, stride=stride) # (1, C*patchsize^2, H*W)
            
            pure_noise=.0
            cov = (1-pure_noise) * (D_train_resized-mu).view(N_imgs, C * H_resized * W_resized).T @ (D_train_resized-mu).view(N_imgs, C * H_resized * W_resized) / N_imgs +pure_noise *torch.eye(C * H_resized * W_resized, device=device) # CHW, CHW
            L = torch.linalg.cholesky(cov + 1e-4 * torch.eye(cov.shape[0], device=cov.device))
            

            pos_patches = extract_centered_patches(torch.arange(C*H_resized*W_resized, device=device).view(1, C, H_resized, W_resized)*1.,patchsize,stride=stride) # 1 , c*patchsize**2, HW
            pos_patches = pos_patches.long()[0] # (C*patchsize^2, H*W)
            rows = pos_patches.T[:, :, None]
            cols = pos_patches.T[:, None, :]
            local_cov = cov[rows, cols] # HW, c*patchsize**2, c*patchsize**2
        
            eigvals, eigvecs = torch.linalg.eigh(local_cov)
        
            thres = 0.1
            #print((eigvals>thres).sum().item())
            eigvals = (eigvals+(eigvals<thres).float())**-1 * (eigvals>thres).float()
            Lambda = torch.diag_embed(eigvals)
            
            #local_cov_inv = torch.linalg.inv(local_cov+ 1e-6 * torch.eye(local_cov.size(1), device=cov.device).unsqueeze(0)) # (C*patchsize^2, C*patchsize^2)
            local_cov_inv = eigvecs @ Lambda @ eigvecs.permute(0, 2, 1)
            
           
            
            
            Z = extract_centered_patches(D_train_resized, patchsize, stride=stride) # (N_imgs, C*patchsize^2, H*W)
            
            x_noise = (L @ torch.randn(C*H*W, device=device)).view(1, C, H, W) + mu
            x_n1 = x_noise
            times = make_times(N, schedule, t0=t0)
            
            saved_steps[0] = x_n1.cpu() if save_immediatly_to_cpu else x_n1.clone()
            
        else: 
            x_n1 = F.interpolate(x_n1, size=(H_resized, W_resized), mode="bicubic").to(device)
            
            t0 = 1. - renoise_factor
            x_n1 = x_n1 * t0 + torch.randn(x_n1.shape, device=device) * (1. - t0)

            times = make_times(int(N * renoise_factor), schedule, t0=t0) 

            Z = extract_centered_patches(D_train_resized, patchsize, stride=stride) # (N_imgs, C*patchsize^2, H*W)
        for it in tqdm(range(times.shape[0]-1), desc=f"Scale {s}/{scales-1}"):
            step += 1
            t = times[it]
            delta_t = times[it + 1] - t 
            
            if s==0: 
                # Extraction de patches
                x_patches = extract_centered_patches(x_n1, patchsize, stride=stride).to(device)
                
                
                # Calcul des distances et poids
                diff = (x_patches - Z * t)/(1-t)  - mu_patches   # (N_imgs, C*patchsize^2, H*W)
                diff_t = diff.permute(0, 2, 1)          # (N_imgs, H*W, C*patchsize^2)
                tmp = torch.bmm(diff_t.permute(1,0,2) , local_cov_inv).permute(1,0,2)
                maha = (tmp * diff_t).sum(-1)           # (N_imgs, H*W)
                w = torch.softmax(-maha / 2, dim=0)
                #print(f"max weight average: {w.max(dim=0).values.mean().item():.4f}")
            else :
                # Extraction de patches
                x_patches = extract_centered_patches(x_n1, patchsize, stride=stride).to(device)
                
                # Calcul des distances et poids
                dists = torch.sum((x_patches - Z * t)**2, dim=1)
                w = torch.softmax(-dists / (2 * ((1 - t) ** 2)), dim=0)
            
            v = ((Z - x_patches) * w.unsqueeze(1)).sum(0, keepdim=True) / (1 - t)
            x_patches_updated = x_patches + v * delta_t

            # Reconstruction de l'image
            x_n1 = weighted_patch_average(
                x_patches_updated, patchsize, C, H_resized, W_resized,
                spatial_weights=spatial_weights, device=device, stride=stride
            )
            
            saved_steps[step] = x_n1.cpu() if save_immediatly_to_cpu else x_n1.clone()
    
    # Batch convert all to CPU at the end if not saving immediately
    if not save_immediatly_to_cpu:
        saved_steps = [img.cpu() for img in saved_steps if img is not None]
    
    return saved_steps

def load_multi_res_tensors(params:dict, scales:int=2, device='cpu') -> list:
    """
    Scales = number of times (- the initial scale) the tensor is "upscaled" (retrieved data from the dataset).
    For example, scales = 3 with initial size of 32x32 would give 3 tensors of size : 32x32, 64x64, 128x128 (initial size + 2 upscalings = 3 scales)
    """
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_tensor_loader import load_data_to_tensor
    
    force_reload_tensor = False
    tensors = []
    
    for i in range(scales):
        params["image_size"] = params["image_size"] * 2 if i > 0 else params["image_size"]
        
        data_dir = "./data"
        torch.cuda.empty_cache()
        data_tensor = load_data_to_tensor(None, data_dir=data_dir, dataset_loading_parameters=params, force_reload_tensor=force_reload_tensor)
        data_tensor = data_tensor.to(device)
        
        tensors.append(data_tensor)
        print(f"Scale {i} loaded")
        
    return tensors

def get_image_tensor_attributes():
    """
    Returns C, H, W
    """
    

def imgs_to_gif(imgs):
    from PIL import Image
    final_size = imgs[-1].shape[-2:]
    
    resized_imgs = [F.interpolate(t, size=final_size, mode="nearest") for t in imgs if t is not None]
    
    np_imgs = [np.uint8(np.clip((img.permute(0, 2, 3, 1).numpy()[0] + 1) / 2, 0, 1) * 255) for img in resized_imgs]
    im_list = [Image.fromarray(np_img, mode='RGB') for np_img in np_imgs]
    im_list[-1].save("out.gif", save_all=True, append_images=im_list[1:], duration=1, loop=0)

def tensor_to_img(t):
    return ((t.permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)

patchsize = 7
stride = patchsize//2
mode = "gaussian"
USE_GPU = True

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    #
    from algorithms.novelty import *
    torch.cuda.empty_cache()
    
    device = "cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu"
    
    dataset_loading_parameters = {
        "data_set_name" : "c",
        "num_samples" : 400,
        "target_labels" : [],
        "image_size" : 32,    
    }
    
    multi_res_tensors = load_multi_res_tensors(dataset_loading_parameters, 3, device)

    seed = torch.randint(0,99999999,(1,1))
    a5 = algo8(
        multi_res_tensors, patchsize=patchsize, stride=stride, 
        N=50, device=device, mask_weight_type=mode, schedule='linear', 
        renoise_factor=0.2, save_immediatly_to_cpu=False, seed=seed
        )
    print(a5[-1].shape)

    a5_clean = [img for img in a5 if img is not None]
    
    # Calculate spatial weights for the comparison, make same channel count as last upscaled image
    comp_channels = multi_res_tensors[-1].shape[1] 
    spatial_weights_for_comp = get_spatial_weights(
        patchsize=patchsize, 
        C=comp_channels, 
        mode=mode,
        device=device
    )[1]

    comparison = compare_ref_stack(
        a5_clean[-1].squeeze(0).to(device), 
        multi_res_tensors[-1].to(device),
        threshold=0.2, 
        distance_gradient=True,
        spatial_weights=spatial_weights_for_comp
    )
    
    
    
    plt.tight_layout()
    
    plt.figure(1)
    plt.title("Generated Image")
    plt.imshow(tensor_to_numpy_img((a5_clean[-1].squeeze(0) + 1) / 2))
    plt.imsave(f"results/algo_9_conv_{seed.item()}.png",tensor_to_numpy_img((a5_clean[-1].squeeze(0)+1)/2))
    
    plt.figure(2)
    plt.title("Patch Regions")
    plt.imshow(tensor_to_numpy_img(comparison[0]))
    plt.imsave(f"results/algo_9_conv_mask_regions_{seed.item()}.png",tensor_to_numpy_img(comparison[0]))
    
    plt.figure(3)
    plt.title("Mosaic View of Patches")
    plt.imshow(tensor_to_numpy_img((comparison[1]+1)/2))
    imgs_to_gif(a5_clean)
    #plt.imsave("poisson_editing/im2.png",tensor_to_numpy_img((comparison[1]+1)/2))
    plt.show()