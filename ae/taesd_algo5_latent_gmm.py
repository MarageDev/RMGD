import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


from gaussian_mixture_noise import get_gmm_noise_gpu
from ae.taesd.taesd import TAESD

PATCHSIZE = 10
STRIDE = 4 #todo
INITIALISE_WITH_LATENT_SPACE_TENSOR = True
SEED = None
ITERATIONS = 25

@torch.no_grad()
def extract_centered_patches(img, patchsize):
    """
    Extrait les patches centrés pour chaque pixel de l'image.
    """
    pad = patchsize // 2
    #img_padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    img_padded = img
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

    fold_layer = nn.Fold((W, H), kernel_size=patchsize, dilation=1, padding=0, stride=STRIDE)
    #fold_layer = nn.Fold((W, H), kernel_size=patchsize, dilation=1, padding=patchsize//2, stride=STRIDE)
    
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
def algo5(D_train, initial_noise,patchsize=3, N=50, schedule='linear', device='cpu', mask_weight_type="standard",seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    D_train = D_train.to(device)
    N_imgs, C, H, W = D_train.shape
    
    # Extract patches from training data
    Z = extract_centered_patches(D_train, patchsize)  # (N_imgs, C*patchsize^2, H*W)
    
    # Initialize with Gaussian noise
    #x_noise = torch.randn(1, C, H, W, device=device)
    #x_noise = get_gmm_noise(D_train.cpu()).to(device).unsqueeze(0)
    x_noise = initial_noise
    
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


@torch.no_grad()
def transform_to_latent_space(tensors, taesd:TAESD, dev):
    """
    Output : N samples, 4, H/(2^upscaling number), W/(2^upscaling number)
    """
    
    from PIL import Image
    import torchvision.transforms.functional as TF

    encoded_tensor = taesd.encoder(tensors[0].unsqueeze(0).to(dev))
    cat_tensor = torch.zeros(tensors.shape[0],*encoded_tensor.shape[1:]).to(dev)
    cat_tensor[0] = encoded_tensor
    for i in tqdm(range(1,tensors.shape[0]),"Encoding tensors"):
        encoded_tensor = taesd.encoder(tensors[i].unsqueeze(0).to(dev))
        cat_tensor[i] = encoded_tensor
    return cat_tensor

def transform_to_latent_space_tensor(tensor, taesd:TAESD, dev):
    import torchvision.transforms.functional as TF

    encoded_tensor = taesd.encoder(tensor.to(dev))

    return encoded_tensor

def imgs_to_gif(path="out.gif",imgs=None):
    from PIL import Image
    
    np_imgs = [np.uint8(np.clip(img.permute(0, 2, 3, 1).detach().numpy()[0] , 0, 1) * 255) for img in imgs]
    im_list = [Image.fromarray(np_img, mode='RGB') for np_img in np_imgs]
    im_list = im_list + [im_list[-1]] * 10
    im_list[0].save(path, save_all=True, append_images=im_list[1:], duration=5, loop=0)


def imgs_to_gif_encode(imgs=None, device="cuda:0"):
    from PIL import Image
    
    imgs_encode_to_rgb = [ TAESD.scale_latents(i.to(device))[0][:3].permute(1,2,0).detach().cpu().numpy() for i in imgs]

    np_imgs = [np.uint8(img * 255) for img in imgs_encode_to_rgb]
    im_list = [Image.fromarray(np_img, mode='RGB') for np_img in np_imgs]
    im_list = im_list + [im_list[-1]] * 100
    im_list[0].save("out_a5_enc.gif", save_all=True, append_images=im_list[1:], duration=1, loop=0)

def decode_tensor(tensor:torch.TensorType, taesd:TAESD):
    return taesd.decoder(tensor).clamp(0, 1)

@torch.no_grad()
def main():
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    
    plt.rcdefaults()
    #import matplotlib as mpl
    #mpl.rcParams["figure.dpi"] = 100
    from algorithms.novelty import compare_ref_stack, tensor_to_numpy_img
    from saver_loader import ls_with_cache_file_tensor, ls_with_cache_file_tensor_dataset, save_dict_hash

    torch.cuda.empty_cache()
    device = "cpu" if False else "cuda:0"
    dataset_loading_parameters = {
        "data_set_name" : "chq",
        "num_samples" : 20,
        "target_labels" : [],
        "image_size" : 512,    
        "normalize" : False,
    }
    
    hash = save_dict_hash(dataset_loading_parameters)
    REWRITE_ALL = True
    data_tensor = ls_with_cache_file_tensor_dataset(search_dir="./data/cached_tensors/dataset", data_dir="./data", 
                                                    dataset_parameters=dataset_loading_parameters, force_rewrite=REWRITE_ALL,device=device)
    
    taesd = TAESD(*["ae/taesd/taesd_encoder.pth", "ae/taesd/taesd_decoder.pth"]).to(device)
    
    encoded_tensor = ls_with_cache_file_tensor(search_dir="./data/cached_tensors/taesd_encoded_dataset", 
                                               preprocess_tensor_function= lambda x : transform_to_latent_space(x,taesd=taesd, dev=device), 
                                               tensor_to_save=data_tensor, device=device, 
                                               requirements=[hash], force_rewrite=REWRITE_ALL)
    #save_or_load_if_hash_encoded(data_tensor, hash=_hash, device=device)

    x_1 = None
    pca_gmm_tensor = torch.load("./data/cached_tensors/gmm/celeba_512_gmm.pt",map_location=device)
    if INITIALISE_WITH_LATENT_SPACE_TENSOR : 
        
        #b=transform_to_latent_space(get_gmm_noise_gpu(data_tensor.cpu(),16, SEED).to(device).unsqueeze(0),taesd=taesd, dev=device)
        b=transform_to_latent_space(pca_gmm_tensor.unsqueeze(0),taesd=taesd, dev=device)
        
        x_1 = b
    else:
        #a=get_gmm_noise_gpu(encoded_tensor.squeeze(0).cpu(),None, seed=SEED).to(device).unsqueeze(0)
        a=pca_gmm_tensor.unsqueeze(0)
        x_1 = a

    a5 = algo5(encoded_tensor, initial_noise=x_1,patchsize=PATCHSIZE, N=ITERATIONS, device=device, mask_weight_type="",schedule='linear', seed=SEED)
    torch.cuda.empty_cache()
    image_dec = decode_tensor(a5[-1].to(device), taesd).cpu()

    imgs_to_gif("out_a5_latent_gmm.gif",[decode_tensor(i.to(device), taesd).cpu() for i in a5])
    imgs_to_gif_encode(a5)

    
    plt.figure(0)
    plt.axis('off')
    plt.imshow(TF.to_pil_image(image_dec[0].detach().clamp(0,1)))
    
    comparison = compare_ref_stack(
        image_dec.squeeze(0).to(device), 
        data_tensor.to(device),
        threshold=0.2, 
        distance_gradient=True,
        spatial_weights=None,
        smooth_kernel=3
    )
    
    plt.figure(1)
    plt.axis('off')
    plt.title("Patch Regions")
    plt.imshow(tensor_to_numpy_img(comparison[0].detach()))
    
    plt.figure(6 )
    plt.axis('off')
    plt.title("Mosaic View of Patches")
    plt.imshow(tensor_to_numpy_img(comparison[1].detach()))
    plt.tight_layout()
    
    plt.show()
    
if __name__ == "__main__":
    main()
