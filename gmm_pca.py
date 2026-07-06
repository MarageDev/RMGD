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

def get_pca_gmm_noise_gpu(image_batch_tensor, pca_A_dim=128, gmm_components=25, device="cuda:0"):

    flattened = image_batch_tensor.view(image_batch_tensor.shape[0], -1)
    mean = flattened.mean(dim=0, keepdim=True).to(device)
    
    # PCA to pca_A_dim dimensions (removes high frequencies with low rank)
    
    A, S, V = torch.pca_lowrank(flattened - mean, q=pca_A_dim, center=True,niter=5) # remove average, in case of (center data)
    pca_features = (flattened - mean) @ V # project into q directions in the pca basis

    # Fit GMM on PCA
    gmm = GaussianMixtureGPU(n_components=gmm_components, n_features=pca_A_dim, max_iter=100,reg_covar=1e-3, n_init=20, device=device)
    gmm.fit(pca_features)
    
    # GMM with X samples
    gmm_samples = gmm.sample(1)[0]
    gmm_samples = torch.as_tensor(gmm_samples, device=device, dtype=torch.float32)
    selected_pca_sample = gmm_samples.unsqueeze(0)

    # Inverse PCA and reshape back to image
    reconstructed = (selected_pca_sample @ V.t()) + mean

    return reconstructed.view(*image_batch_tensor.shape[1:])

if __name__ == "__main__" :
    # from saver_loader import ls
    import os
    import dataset_loader

    dataset_loading_parameters = {
        "data_set_name" : "o",
        "num_samples" : 500,
        "target_labels" : [],
        "image_size" : 512,    
        "normalize" : False,
    }
    #image_batch_tensor = ls_with_cache_file_tensor_dataset(dataset_parameters=dataset_loading_parameters,device="cuda:0",force_rewrite=True)

    image_batch_tensor = dataset_loader.load_parquet(root="./data/celebahq", num_samples=50, image_size=64, normalize=False)
    if not image_batch_tensor.is_cuda:
        image_batch_tensor = image_batch_tensor.cuda()

    generated_noise_tensor = get_pca_gmm_noise_gpu(
        image_batch_tensor, 
    )
    os.makedirs("./data/cached_tensors/gmm", exist_ok=True)
    torch.save(generated_noise_tensor, "./data/cached_tensors/gmm/celeba_512_gmm.pt")
    plt.imshow(TF.to_pil_image(generated_noise_tensor.cpu()))
    plt.axis("off")
    plt.show()