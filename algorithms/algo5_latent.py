# Based on algorithm 5 but in latent space of TAESD (latest algortihm)

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
        sig_pix = 1.0 * half_win_size

        # arange creates the spatial spread
        dx = torch.linspace(-half_win_size, half_win_size, steps=patchsize, device=device)
        w1d = torch.exp(-(dx / sig_pix) ** 2 / 2.0)

        # Create 2D weight and adapt to 3 color channels
        w2d = w1d[:, None] * w1d[None, :]
        w = w2d.repeat(C, 1, 1).reshape(-1)
        w = w / w.sum()  # normalize
        w = w.unsqueeze(0).unsqueeze(-1)        
        
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
    
    # Extract patches from training data
    Z = extract_centered_patches(clean_batch_tensor, patchsize, stride=stride)
    x_noise = initialisation_tensor 
    
    # test JUJU todo
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
        # TEST JUJU
        #x_patches_updated = Z[1:2]
        #x_patches_updated *= 1/0.18  
        
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
@torch.no_grad()
def main():

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    plt.rcdefaults()
    plt.rcParams.update(plt.rcParamsDefault)

    from algorithms.novelty import compare_ref_stack, tensor_to_numpy_img
    from others.saver_loader import ls_with_cache_file_tensor, ls_with_cache_file_tensor_dataset, save_dict_hash
    from algorithms.algorithm_PCAGMM_latent import PCAGMMFaceGenerator
    torch.cuda.empty_cache()
    device = "cpu" if False else "cuda:0"

    #   LOAD TENSORS & LOAD INITIALISATION TENSOR (PCA & GMM IN LATENT SPACE)
    ##########################################################################
    taesd = TAESD(*["ae/taesd/taesd_encoder.pth", "ae/taesd/taesd_decoder.pth"]).to(device)
    
    encoded_tensor = torch.load("data/cached_tensors/taesd_encoded_dataset/fully_encoded_celebahq_dataset.pt", weights_only=False, map_location="cuda:0")[0:ALGO_SAMPLES]

    pcagmm_generator = PCAGMMFaceGenerator.load(
        #"./data/cached_tensors/pcagmm/tensor_pcagm_188853386220802287e4d07d96d335828a8fc58ab191c1ee410ec1e48bb0e11c_fbf1f9c98eefe3aae0e9023e99d4406820fffa2b212c2e341f2034296761be54_188853386220802287e4d07d96d335828a8fc58ab191c1ee410ec1e48bb0e11c.pt", # 256 pca, 1000 gmm
        "./data/cached_tensors/pcagmm/tensor_pcagm_188853386220802287e4d07d96d335828a8fc58ab191c1ee410ec1e48bb0e11c_05a5d1072c9fd93d4050e86c23df981b77820dad47ea26fb0b90e127b382881b_188853386220802287e4d07d96d335828a8fc58ab191c1ee410ec1e48bb0e11c.pt", # 512 pca, 10 gmm
        device="cpu").sample(1, return_average=False, seed = SEED).to(device)
    x_1 = pcagmm_generator
    #nb =  1
    #n = INIT_PC_FACTOR
    #x_1= pcagmm_generator.sample_pc(INIT_PC, np.interp(n,[-1.,1.],[*pcagmm_generator.get_pc_distribution(encoded_tensor, nb)[:-1]]),return_average=False)
    #x_1 = pcagmm_generator.sample_multiple_pcs(pc_weights={5 : -60,3 : 82, 8:-63, 18:38, 39:-18, 7:-59, 0:80},return_average=False)
    torch.cuda.empty_cache()
    
    print('\x1b[6;30;42m' + 'All tensors loaded with succcess' + '\x1b[0m')
    
    
    
    #   RUN THE ALGORITHM
    #####################
    
    results_lat = ALGO_LATENT(encoded_tensor, initialisation_tensor=x_1,patchsize=PATCHSIZE, N=ITERATIONS, device=device, mask_weight_type="linear",schedule='cosine', seed=SEED)
    torch.cuda.empty_cache()
    
    #   PREPARE FOR DISPLAY AND OUTPUT
    ##################################
    
    final_result_lat = results_lat[-1]
    final_result_rgb = decode_tensor(final_result_lat.to(device), taesd)
    
    # GIF CREATION (SLOW TO GENERATE)
    if OUTPUT_RGB_GIF : imgs_to_gif("out_a5_latent_gmm.gif",[decode_tensor(i.to(device), taesd).cpu() for i in results_lat])
    if OUTPUT_LATENT_GIF : imgs_to_gif_encode(results_lat)

    # MATPLOTLIB DISPLAY
    fig = plt.figure(layout="constrained", dpi=100)
    axs = fig.subplot_mosaic([
        ['synth', 'init',      'mosaic',       "predom_mosaic",    "coi_mosaic"],
        ['sampled_init', 'patches',   'novelty_dist', "predom",           "coi"       ]
    ])
   
    quick_decode = lambda x : decode_tensor(x.unsqueeze(0).to(device), taesd).cpu()
    quick_image_decode = lambda x : tensor_to_numpy_img(quick_decode(x)[0])
    
    # print(final_result_lat.shape, final_result_rgb.shape)
    axs["synth"].set_title("Result")
    axs["synth"].imshow(TF.to_pil_image(final_result_rgb[0].cpu().detach().clamp(0,1)))
    axs["synth"].axis("off")

    axs["sampled_init"].set_title("Sampled Init")
    axs["sampled_init"].imshow(quick_image_decode(x_1[0].cpu()))
    axs["sampled_init"].axis("off")
    
    axs["init"].set_title("Initialisation")
    axs["init"].imshow(quick_image_decode(results_lat[0][0].cpu()))
    axs["init"].axis("off")

    comparison = compare_ref_stack(
        final_result_lat[0].to(device), 
        encoded_tensor.to(device),
        threshold=0.9, 
        distance_gradient=True,
        spatial_weights=None,
        smooth_kernel=PATCHSIZE+1
    )
    
    

    

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