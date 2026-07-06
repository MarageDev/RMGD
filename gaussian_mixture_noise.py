import numpy as np

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F


from time import perf_counter

from tgmm import GaussianMixture as GaussianMixtureGPU
from sklearn.mixture import GaussianMixture

def get_gmm_noise_cpu(image_batch_tensor:torch.Tensor, reduced_size = None):
    initial_size = image_batch_tensor.shape[-2:]
    if reduced_size is not None : 
        image_batch_tensor = F.interpolate(image_batch_tensor, size=reduced_size, mode="bicubic")
    flattened_images = image_batch_tensor.view(image_batch_tensor.shape[0], -1).detach().numpy()

    # fit gmm
    t = perf_counter()
    gmm = GaussianMixture(n_components=1,max_iter=25,reg_covar=1e-2)#,random_state=1)
    gmm.fit(flattened_images)
    print("gmm fitting : ",perf_counter() - t)

    # Generate new images by sampling from gmm
    t = perf_counter()
    generated_samples = gmm.sample(1)[0]  # (num_generated,samples(channels*height*width))

    image_shape = image_batch_tensor.shape[1:]  # (3, 32, 32)
    generated_image = torch.tensor(generated_samples).view(*image_shape).float() # Reshape back to (channels, height, width)
    if reduced_size is not None : 
        generated_image = F.interpolate(generated_image.unsqueeze(0), size=initial_size, mode="bicubic").squeeze(0)
    return generated_image


def get_gmm_noise_gpu(image_batch_tensor:torch.Tensor, reduced_size = None, seed=None):
    initial_size = image_batch_tensor.shape[-2:]
    if reduced_size is not None : 
        image_batch_tensor = F.interpolate(image_batch_tensor, size=reduced_size, mode="bicubic")
    
    flattened_images = image_batch_tensor.view(image_batch_tensor.shape[0], -1) #  (image_batch_size, height*widht*channels)

    # fit gmm
    t = perf_counter()
    gmm = GaussianMixtureGPU(n_components=1,max_iter=25,reg_covar=1e-3, device="cuda",random_state=seed)#,random_state=1)
    gmm.fit(flattened_images)
    print("gmm fitting : ",perf_counter() - t)

    # Generate new images by sampling from gmm
    t = perf_counter()
    generated_samples = gmm.sample(1)[0]  # (num_generated,samples(channels*height*width))

    image_shape = image_batch_tensor.shape[1:]  # (3, 32, 32)
    generated_image = torch.tensor(generated_samples).view(*image_shape).float() # Reshape back to (channels, height, width)
    if reduced_size is not None : 
        generated_image = F.interpolate(generated_image.unsqueeze(0), size=initial_size, mode="bicubic").squeeze(0)
    return generated_image

if __name__ == "__main__" :
    from data_tensor_loader import load_data_to_tensor
    import matplotlib.pyplot as plt
    dataset_loading_parameters = {
        "data_set_name" : "c",
        "num_samples" : 100,
        "target_labels" : [0],
        "image_size" : 32,    
        "normalize" : False,
    }
    image_batch_tensor, _ = load_data_to_tensor(None,dataset_loading_parameters=dataset_loading_parameters)

    # Visualize generated images
    plt.imshow(TF.to_pil_image(get_gmm_noise_gpu(image_batch_tensor)))
    plt.axis("off")
    plt.show()