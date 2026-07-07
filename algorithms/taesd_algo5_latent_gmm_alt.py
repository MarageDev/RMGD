# Based on algorithm 5 but in latent space of TAESD

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

PATCHSIZE = 8
STRIDE = PATCHSIZE//2
ITERATIONS = 30

INITIAL_NOISE_FACTOR = 0.5

SEED = None


OUTPUT_RGB_GIF = True
OUTPUT_LATENT_GIF = False

REWRITE_ALL_TENSOR_FILES = False

dataset_loading_parameters = {
    "data_set_name" : "chq",
    "num_samples" : 100,
    "target_labels" : [],
    "image_size" : 512,    
    "normalize" : False,
}

PCAGMM_SETTINGS = {
    "PCA_dim": 256,
    "GMM_comp": 5,
    "PCA_ITER":100, 
    "GMM_INIT_ITER":5
}

@torch.no_grad()
def extract_centered_patches(img, patchsize):
    """
    Extrait les patches centrés pour chaque pixel de l'image.
    """
    pad = patchsize // 2
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
def ALGO_LATENT(clean_batch_tensor, initialisation_tensor, patchsize=3, N=50, schedule='linear', device='cpu', mask_weight_type="standard", seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    clean_batch_tensor = clean_batch_tensor.to(device)
    N_imgs, C, H, W = clean_batch_tensor.shape
    
    # Extract patches from training data
    Z = extract_centered_patches(clean_batch_tensor, patchsize)
    x_noise = initialisation_tensor 
    
    # test JUJU todo
    x_noise = (1. - INITIAL_NOISE_FACTOR ) * x_noise + INITIAL_NOISE_FACTOR * torch.randn_like(x_noise)
    #t0 = (1. - INITIAL_NOISE_FACTOR )
    t0=0.
    
    x_n1 = x_noise.clone()
    
    saved_steps = [x_n1.cpu()]
    
    times = make_times(N, schedule, t0)
    
    for it in tqdm(range(N)):
        t = times[it]
        delta_t = times[it + 1] - t 

        x_patches = extract_centered_patches(x_n1, patchsize) # (1, C*patchsize^2, H*W) 
        
        dists = torch.sum((x_patches - Z * t)**2, dim=1) # (N_imgs, H*W)

        w = torch.softmax(-dists / (2 * ((1. - t) ** 2)), dim=0) # (N_imgs, H*W)  

        v = ((Z - x_patches) * w.unsqueeze(1)).sum(0, keepdim=True) / (1. - t)
        # Z : (N_imgs, C*patchsize^2, H*W)  // x_patches (1, C*patchsize^2, H*W) // w (N_imgs, H*W)
        
        x_patches_updated = x_patches + v * delta_t
        # TEST JUJU
        #x_patches_updated = Z[1:2]
        #x_patches_updated *= 1/0.18  
        
        x_n1 = weighted_patch_average(x_patches_updated, patchsize, C, H, W, mode=mask_weight_type, device=device)
        
        
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
@torch.no_grad()
def main():

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    plt.rcdefaults()
    from algorithms.novelty import compare_ref_stack, tensor_to_numpy_img
    from saver_loader import ls_with_cache_file_tensor, ls_with_cache_file_tensor_dataset, save_dict_hash
    from gmm_pca_rework import PCAGMMFaceGenerator, quick_create_model
    torch.cuda.empty_cache()
    device = "cpu" if False else "cuda:0"
    
    hash = save_dict_hash(dataset_loading_parameters)
    
    #   LOAD TENSORS
    ################
    data_tensor = ls_with_cache_file_tensor_dataset(search_dir="./data/cached_tensors/dataset", 
                                                    data_dir="./data", force_rewrite=REWRITE_ALL_TENSOR_FILES,
                                                    dataset_parameters=dataset_loading_parameters, 
                                                    requirements=[],device=device)
    
    taesd = TAESD(*["ae/taesd/taesd_encoder.pth", "ae/taesd/taesd_decoder.pth"]).to(device)
    encoded_tensor = ls_with_cache_file_tensor(search_dir="./data/cached_tensors/taesd_encoded_dataset", 
                                               preprocess_tensor_function= lambda x : transform_to_latent_space(x,taesd=taesd, dev=device), 
                                               tensor_to_save=data_tensor, device=device, 
                                               requirements=["encoded",hash], force_rewrite=REWRITE_ALL_TENSOR_FILES)
    print("Clean batches tensor loaded")
    
    
    
    
    #   LOAD INITIALISATION TENSOR (PCA & GMM IN LATENT SPACE)
    ##########################################################

    #pca_gmm_tensor = torch.load("./data/cached_tensors/gmm/celebahq_512_lat2.pt",map_location=device)
    pca_gmm_tensor = quick_create_model([hash, encoded_tensor], PCAGMM_SETTINGS, seed=SEED).sample(1, return_average=False, seed = SEED)
    x_1=pca_gmm_tensor
    
    print("PCAGMM model created and sampled")
    
    
    
    print('\x1b[6;30;42m' + 'All tensors loaded with succcess' + '\x1b[0m')
    
    
    
    #   RUN THE ALGORITHM
    #####################
    
    results_lat = ALGO_LATENT(encoded_tensor, initialisation_tensor=x_1,patchsize=PATCHSIZE, N=ITERATIONS, device=device, mask_weight_type="",schedule='linear', seed=SEED)
    torch.cuda.empty_cache()
    
    #   PREPARE FOR DISPLAY AND OUTPUT
    ##################################
    
    final_result_lat = results_lat[-1]
    final_result_rgb = decode_tensor(final_result_lat.to(device), taesd)
    
    # GIF CREATION (SLOW TO GENERATE)
    if OUTPUT_RGB_GIF : imgs_to_gif("out_a5_latent_gmm.gif",[decode_tensor(i.to(device), taesd).cpu() for i in results_lat])
    if OUTPUT_LATENT_GIF : imgs_to_gif_encode(results_lat)

    # MATPLOTLIB DISPLAY
    fig = plt.figure(layout="constrained")
    axs = fig.subplot_mosaic([
        ['synth', 'synth', 'init',      'mosaic',       "predom_mosaic",    "coi_mosaic"],
        ['synth', 'synth', 'patches',   'novelty_dist', "predom",           "coi"       ]
    ])
    # print(final_result_lat.shape, final_result_rgb.shape)
    axs["synth"].set_title("Result")
    axs["synth"].imshow(TF.to_pil_image(final_result_rgb[0].cpu().detach().clamp(0,1)))
    axs["synth"].axis("off")
    
    axs["init"].set_title("Initialisation")
    axs["init"].imshow(TF.to_pil_image(decode_tensor(x_1.to(device), taesd).cpu()[0].detach().clamp(0,1)))
    axs["init"].axis("off")

    comparison = compare_ref_stack(
        final_result_lat[0].to(device), 
        encoded_tensor.to(device),
        threshold=0.2, 
        distance_gradient=True,
        spatial_weights=None,
        smooth_kernel=PATCHSIZE+1
    )
    
    def quick_decode(x):
        return decode_tensor(x.unsqueeze(0).to(device), taesd).cpu()

    quick_image_decode = lambda x : tensor_to_numpy_img(quick_decode(x)[0])

    axs["patches"].set_title("Patch Regions")
    axs["patches"].imshow(tensor_to_numpy_img(comparison[0]))
    axs["patches"].axis("off")
    
    axs["mosaic"].set_title("Mosaic View of Patches")
    axs["mosaic"].imshow(quick_image_decode(comparison[1]))
    axs["mosaic"].axis("off")

    axs["predom"].set_title("Predominant")
    axs["predom"].imshow(quick_image_decode(comparison[2]))
    axs["predom"].axis("off")
    
    axs["predom_mosaic"].set_title("Predominant mosaic")
    axs["predom_mosaic"].imshow(quick_image_decode(comparison[3]))
    axs["predom_mosaic"].axis("off")
    
    axs["coi"].set_title("COI")
    axs["coi"].imshow(quick_image_decode(comparison[4]))
    axs["coi"].axis("off")
    
    axs["coi_mosaic"].set_title("COI mosaic")
    axs["coi_mosaic"].imshow(quick_image_decode(comparison[5]))
    axs["coi_mosaic"].axis("off")
    
    axs["novelty_dist"].set_title("Novelty dist")
    axs["novelty_dist"].imshow(tensor_to_numpy_img(comparison[6]))
    axs["novelty_dist"].axis("off")
    
    plt.imsave("novelty_test.jpg",TF.to_pil_image(final_result_rgb[0].cpu().detach().clamp(0,1)))
    
    plt.show()
    
if __name__ == "__main__":
    
    main()