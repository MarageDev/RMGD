import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from saver_loader import ls_with_cache_file_tensor_dataset

import torch
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from time import perf_counter

from tgmm import GaussianMixture as GaussianMixtureGPU
import os
import dataset_loader

def get_pca_gmm_noise_gpu(image_batch_tensor, pca_A_dim=32, gmm_components=1, device="cuda:0"):

    flattened = image_batch_tensor.view(image_batch_tensor.shape[0], -1)
    mean = flattened.mean(dim=0, keepdim=True).to(device)
    
    # PCA to pca_A_dim dimensions (removes high frequencies with low rank)
    
    U, S, V = torch.pca_lowrank(flattened - mean, q=pca_A_dim, center=True,niter=5) # remove average, in case of (center data)
    pca_features = (flattened - mean) @ V # project into q directions in the pca basis

    # Fit GMM on PCA
    gmm = GaussianMixtureGPU(n_components=gmm_components, max_iter=300,reg_covar=1e-2, n_init=20, device=device)
    gmm.fit(pca_features)
    
    # GMM with X samples
    gmm_samples = gmm.sample(1)[0]
    gmm_samples = torch.as_tensor(gmm_samples, device=device, dtype=torch.float32)
    selected_pca_sample = gmm_samples.unsqueeze(0)

    # Inverse PCA and reshape back to image
    reconstructed = (selected_pca_sample @ V.t()) + mean

    return reconstructed.view(*image_batch_tensor.shape[1:])


from ae.taesd.taesd import TAESD
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

if __name__ == "__main__":

    from saver_loader import ls_with_cache_file_tensor, ls_with_cache_file_tensor_dataset, save_dict_hash

    torch.cuda.empty_cache()
    device = "cpu" if False else "cuda:0"
    dataset_loading_parameters = {
        "data_set_name" : "chq",
        "num_samples" : 500,
        "target_labels" : [],
        "image_size" : 512,    
        "normalize" : False,
    }
    
    hash = save_dict_hash(dataset_loading_parameters)
    
    data_tensor = ls_with_cache_file_tensor_dataset(search_dir="./data/cached_tensors/dataset", data_dir="./data", 
                                                    dataset_parameters=dataset_loading_parameters, force_rewrite=False,device=device)
    taesd = TAESD(*["ae/taesd/taesd_encoder.pth", "ae/taesd/taesd_decoder.pth"]).to(device)
    encoded_tensor = ls_with_cache_file_tensor(search_dir="./data/cached_tensors/taesd_encoded_dataset", 
                                               preprocess_tensor_function= lambda x : transform_to_latent_space(x,taesd=taesd, dev=device), 
                                               tensor_to_save=data_tensor, device=device, 
                                               requirements=[hash], force_rewrite=True)
    generated_noise_tensor = get_pca_gmm_noise_gpu(
        encoded_tensor, 
    )
    os.makedirs("./data/cached_tensors/gmm", exist_ok=True)
    torch.save(generated_noise_tensor, "./data/cached_tensors/gmm/celebahq_512_lat.pt")
    plt.imshow(TF.to_pil_image(generated_noise_tensor.cpu()))
    plt.axis("off")
    plt.show()
