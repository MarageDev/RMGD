import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def extract_centered_patches(img, patchsize):
    """
    Extrait les patches centrés pour chaque pixel de l'image.
    """
    pad = patchsize // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    return F.unfold(img_padded, kernel_size=patchsize, padding=0, stride=patchsize//2)

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

    fold_layer = nn.Fold((W, H), kernel_size=patchsize, dilation=1, padding=patchsize//2, stride=patchsize//2)
    
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
def algo_cov(D_train, patchsize=3, N=50, schedule='linear', device='cpu', mask_weight_type="standard",seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    D_train = D_train.to(device)


    N_imgs, C, H, W = D_train.shape
    mu = D_train.mean(dim=(0), keepdim=True) # (1, C, H, W)
    pure_noise=0.
    cov = (1-pure_noise) * (D_train-mu).view(N_imgs, C * H * W).T @ (D_train-mu).view(N_imgs, C * H * W) / N_imgs  +pure_noise *torch.eye(C * H * W, device=device) # CHW, CHW

    pos_patches = extract_centered_patches(torch.arange(C*H*W, device=device).view(1, C, H, W)*1.,patchsize) # 1 , c*patchsize**2, HW
    pos_patches = pos_patches.long()[0]  # (C*patchsize^2, H*W)
    rows = pos_patches.T[:, :, None]
    cols = pos_patches.T[:, None, :]
    local_cov = cov[rows, cols] # HW, c*patchsize**2, c*patchsize**2
    local_cov_inv = torch.linalg.inv(local_cov+ 1e-6 * torch.eye(local_cov.size(1), device=cov.device).unsqueeze(0)) # (hw, C*patchsize^2, C*patchsize^2)



    if False:
        import matplotlib.pyplot as plt


        evals_g, evecs_g = torch.linalg.eigh(cov)   # (CHW,), (CHW, CHW)

        v_global = evecs_g[:, -1]                   # (CHW,)

        img_global = v_global.view(C, H, W)

        # fix arbitrary sign for visualization
        img_global = img_global * torch.sign(
            img_global.flatten()[img_global.abs().argmax()]
        )

        plt.figure(figsize=(5,5))
        if C == 1:
            plt.imshow(img_global[0].cpu(), cmap='gray')
        else:
            img_vis = img_global.permute(1,2,0)
            img_vis = img_vis / img_vis.abs().max()
            plt.imshow((img_vis[0].cpu()+1)/2)
        plt.title("Dominant eigenvector of global covariance")
        plt.axis('off')
        plt.show()



        print(local_cov.shape)
        evals, evecs = torch.linalg.eigh(local_cov+ 1e-6 * torch.eye(local_cov.size(1), device=cov.device).unsqueeze(0).repeat(H*W, 1, 1))   # evecs: (HW, D, D)

        # strongest eigenvector
        eigh = evecs[..., -1]                         # (HW, D)

        # optional sign alignment to avoid random flips
        eigh = eigh * torch.sign(eigh[:, :1]).clamp(min=1)



        eigh = eigh.T.unsqueeze(0)                    # (1, D, HW)

        fold_layer = nn.Fold(
            output_size=(H, W),
            kernel_size=patchsize,
            padding=patchsize//2,
            stride=1,
        )

        refold = fold_layer(eigh)                     # (1, C, H, W) if D=C*patchsize²

        ones = torch.ones_like(eigh)
        norm = fold_layer(ones)

        refold = refold / norm.clamp_min(1e-8)

        plt.figure(figsize=(6,6))
        plt.imshow(refold[0].permute(1,2,0).cpu())
        plt.axis('off')
        plt.show()








    L = torch.linalg.cholesky(cov + 1e-4 * torch.eye(cov.shape[0], device=cov.device))

    
    # Extract patches from training data
    Z = extract_centered_patches(D_train, patchsize)  # (N_imgs, C*patchsize^2, H*W)
    
    # Initialize with Gaussian noise
    #x_noise = torch.randn(1, C, H, W, device=device)
        
    x_noise = (L @ torch.randn(C*H*W, device=device)).view(1, C, H, W) +mu
    #x_noise = torch.randn(1, C, H, W, device=device)#!!!!!
    x_n1 = x_noise.clone()
    
    # Save initial noise
    saved_steps = [x_n1.cpu()]
    
    times = make_times(N, schedule, 0)
    
    for it in tqdm(range(N)):
        t = times[it]

        delta_t = times[it + 1] - t 
        

        x_patches = extract_centered_patches(x_n1, patchsize)  # (1, C*patchsize^2, H*W)
        mu_patches = extract_centered_patches(mu, patchsize)  # (1, C*patchsize^2, H*W)
        
        #slice cov
    
        #cov_inv = torch.linalg.inv(local_cov+ 1e-6 * torch.eye(local_cov.size(0), device=cov.device)) # (C*patchsize^2, C*patchsize^2)

        diff = (x_patches - Z * t)/(1-t)  - mu_patches   # (N_imgs, C*patchsize^2, H*W)
        diff_t = diff.permute(0, 2, 1)          # (N_imgs, H*W, C*patchsize^2)
           # (N_imgs, H*W, C*patchsize^2)      HW, CP**2,CP**2   
        tmp = torch.bmm(diff_t.permute(1,0,2) , local_cov_inv).permute(1,0,2)                  # (N_imgs, H*W, C*patchsize^2)
        maha = (tmp * diff_t).sum(-1)           # (N_imgs, H*W)
        w = torch.softmax(-maha / 2, dim=0)
        print(f"max weight average: {w.max(dim=0).values.mean().item():.4f}")   

        #dists = torch.sum((x_patches - Z*t)**2, dim=1)  # (N_imgs, H*W)
        #w = torch.softmax(-dists / (2 * ((1 - t) ** 2)), dim=0)  # (N_imgs, H*W)
        
        v = ((Z - x_patches) * w.unsqueeze(1)).sum(0, keepdim=True) / (1 - t)
        # Z : (N_imgs, C*patchsize^2, H*W)  // x_patches (1, C*patchsize^2, H*W) // w (N_imgs, H*W)
        
        
        x_patches_updated = x_patches + v * delta_t

        x_n1 = weighted_patch_average(x_patches_updated, patchsize, C,H, W,mode=mask_weight_type ,device=device)
        
        saved_steps.append(x_n1.cpu())
    
    return saved_steps


def imgs_to_gif(imgs):
    from PIL import Image
    
    np_imgs = [np.uint8(np.clip((img.permute(0, 2, 3, 1).numpy()[0] + 1) / 2, 0, 1) * 255) for img in imgs]
    im_list = [Image.fromarray(np_img, mode='RGB') for np_img in np_imgs]
    im_list[0].save("out.gif", save_all=True, append_images=im_list[1:], duration=1, loop=0)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    torch.cuda.empty_cache()
    device = "cpu" if False else "cuda:0"
    tensor_file = "tensor_cache_0_24f871758c12de6ba842d73c8c2e3c4361cce6d3d417a6db6dc8e85d1c41b074.pt"
    tensor_path = f"./data/saved_tensors/{tensor_file}"
    data_tensor = torch.load(tensor_path, map_location=device)
    a5 = algo_cov(data_tensor, patchsize=5, N=100, device=device, mask_weight_type="gaussian",schedule='linear', seed=1)
    
    
    
    plt.imsave(f"nifty_alg_5{1}.png",np.clip((a5[0].permute(0, 2, 3, 1).numpy()[0] + 1) / 2, 0, 1))
    plt.imsave(f"nifty_alg_5{2}.png",np.clip((a5[-1].permute(0, 2, 3, 1).numpy()[0] + 1) / 2, 0, 1))
    imgs_to_gif(a5)
    plt.tight_layout()
    plt.imshow(np.clip((a5[-1].permute(0, 2, 3, 1).numpy()[0] + 1) / 2, 0, 1))
    plt.show()
    
