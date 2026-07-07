import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import os
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from time import perf_counter

from data_tensor_loader import load_data_to_tensor
from tgmm import GaussianMixture as GaussianMixtureGPU

class PCAGMMFaceGenerator:
    from typing import Self
    
    def __init__(self, pca_A_dim:int=256, gmm_components:int=25, pca_iter:int = 100, gmm_init_iter:int = 100, seed:int = None,device:str="cuda:0") -> None:
        
        
        self.pca_A_dim = pca_A_dim
        self.gmm_components = gmm_components
        self.device = device
        self.seed = seed
        
        self.pca_iter = 100
        self.gmm_init_iter = 100
        # State vars
        self.mean = None
        self.V = None
        self.gmm = None
        self.image_shape = None
        
        

    def fit(self, encoded_batch_tensor:torch.Tensor) -> None:
        """PCA with GMM on it"""
        print("Fitting model")

        self.image_shape = encoded_batch_tensor.shape[1:] # Save the shape of a single image
        
        flattened = encoded_batch_tensor.view(encoded_batch_tensor.shape[0], -1)
        self.mean = flattened.mean(dim=0, keepdim=True).to(self.device)
        
        # PCA to pca_A_dim dimensions
        U, S, self.V = torch.pca_lowrank(flattened - self.mean, q=self.pca_A_dim, center=True, niter=self.pca_iter) 
        pca_features = (flattened - self.mean) @ self.V

        # Fit GMM on PCA
        self.gmm = GaussianMixtureGPU(
            n_components=self.gmm_components, 
            n_features=self.pca_A_dim, 
            max_iter=1000,
            reg_covar=1e-3, 
            n_init=self.gmm_init_iter, 
            device=self.device,
            random_state=self.seed
        )
        self.gmm.fit(pca_features)
        print("Model fitted")

    def sample(self, num_samples=1, return_average=False, seed=None):
        """
        Function to generate sample from pre-fitted model
        
        Returns a tensor of shape (num_samples, C, H, W)
        """
        if self.gmm is None or self.V is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            self.gmm.random_state = seed
        
        
        
        if return_average:
            reconstructed = self.mean.expand(num_samples, -1) # Returns the global average face of the dataset, duplicated for num_samples
        else:
            gmm_samples = self.gmm.sample(num_samples)[0]  # Sample from GMM
            gmm_samples = torch.as_tensor(gmm_samples, device=self.device, dtype=torch.float32)

            if gmm_samples.ndim == 1: # Ensure 2D shape (num_samples, pca_A_dim) just in case
                gmm_samples = gmm_samples.unsqueeze(0)

            reconstructed = (gmm_samples @ self.V.t()) + self.mean # Inverse PCA

       
        return reconstructed.view(num_samples, *self.image_shape) # Reshape back to image format
    
    def get_save_state(self) -> dict:
        state = {
            'mean': self.mean,
            'V': self.V,
            'gmm': self.gmm,
            'image_shape': self.image_shape,
            'pca_A_dim': self.pca_A_dim,
            'gmm_components': self.gmm_components,
            'pca_iter' : self.pca_iter,
            'gmm_init_iter' : self.gmm_init_iter
        }
        
        return state
    
    def save(self, filepath:str) -> None:
        """Saves the fitted model state"""
        state = {
            'mean': self.mean,
            'V': self.V,
            'gmm': self.gmm,
            'image_shape': self.image_shape,
            'pca_A_dim': self.pca_A_dim,
            'gmm_components': self.gmm_components,
            'pca_iter' : self.pca_iter,
            'gmm_init_iter' : self.gmm_init_iter
        }
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath:str, device:str="cuda:0") -> Self:
        """Loads a model from a file"""
        state = torch.load(filepath, map_location=device, weights_only=False)
        
        # Instantiate class with stored parameters
        generator = cls(
            pca_A_dim=state['pca_A_dim'], 
            gmm_components=state['gmm_components'], 
            pca_iter= state['pca_iter'], 
            gmm_init_iter = state['gmm_init_iter'],
            device=device
        )
        
        # Restore state
        generator.mean = state['mean'].to(device)
        generator.V = state['V'].to(device)
        generator.gmm = state['gmm']
        generator.image_shape = state['image_shape']
        
        return generator

DEFAULT_PCAGMM_SETTINGS = {
    "PCA_dim": 256,
    "GMM_comp": 5,
    "PCA_ITER":100, 
    "GMM_INIT_ITER":100
}

def quick_create_model(fitting_data_tensor:tuple[str,torch.TensorType],settings:dict = {"PCA_dim": 256,"GMM_comp": 5, "PCA_ITER":100, "GMM_INIT_ITER":100},search_dir:str="./data/cached_tensors/pcagmm",save_path:str="./data/cached_tensors/pcagmm", force_rewrite:bool=False, seed:int=None,device:str="cuda:0") -> PCAGMMFaceGenerator:
    """
    Create a PCAGMM Face generator (class PCAGMMFaceGenerator) from either an already existing file matching the settings of the generator entered, or by creating one fitted to `fitting_data_tensor` and saving it.
    
    Fitting data tensor is a tuple of the hash of the data tensor and the tensor itself.
    
    Return a PCAGMMFaceGenerator object.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    import saver_loader as ls
    
    dataset_hash, dataset_tensor = fitting_data_tensor
    pcagmm_hash = ls.save_dict_hash(settings)
    requirements = ["pcagm", dataset_hash, pcagmm_hash] # Search for tensors with "pcagmm", the hash of the fitting data tensor used (number of samples, resolution, dataset used ...) and the hash of the pcagmm settings
    
    # Reuse of the ls.ls_with_cache_file_tensor function
    os.makedirs(search_dir, exist_ok=True)

    
    load_from_cache, existant_file_path = ls.has_corresponding_reqs(search_dir, requirements)
    can_load = not force_rewrite and load_from_cache
    
    data_tensor = None

    if can_load:
        generator = PCAGMMFaceGenerator.load(f"{search_dir}/{existant_file_path}", device="cuda:0")
        generator.seed = seed
        print(f"Loaded PCAGMM model from {search_dir}/{existant_file_path}")
        
        return generator
    else:
        generator = PCAGMMFaceGenerator(pca_A_dim=settings["PCA_dim"], gmm_components=settings["GMM_comp"], pca_iter=settings["PCA_ITER"],gmm_init_iter=["GMM_INIT_ITER"],device="cuda:0")
        generator.seed = seed
        generator.fit(dataset_tensor)
        generator_state = generator.get_save_state()

        tensor_file_name = f"tensor_{"_".join([*requirements])}_{dataset_hash}.pt" 
        torch.save(generator_state, f"{search_dir}/{tensor_file_name}")
        
        print(f"Saved and loaded PCAGMM model at {search_dir}/{tensor_file_name}")
        
        return generator

def one():
    generator = PCAGMMFaceGenerator(pca_A_dim=256, gmm_components=25) # create
    generator.fit(encoded_batch_tensor) # fit model
    generator.save("./data/cached_tensors/gmm/face_model.pt") # savemodel

def one_load():
    # Load the model from disk
    generator = PCAGMMFaceGenerator.load("./data/cached_tensors/gmm/face_model.pt", device="cuda:0")

    # sample instantly without fitting
    new_faces = generator.sample(num_samples=5)

    # or just get the average face
    #avg_face = generator.sample(num_samples=1, return_average=True)

from ae.taesd.taesd import TAESD
@torch.no_grad()
def transform_to_latent_space(tensors, taesd:TAESD, dev):
    from PIL import Image
    from tqdm import tqdm
    import torchvision.transforms.functional as TF

    # Scale latents immediately after encoding
    encoded_tensor = taesd.encoder(tensors[0].unsqueeze(0).to(dev))
    cat_tensor = torch.zeros(tensors.shape[0], *encoded_tensor.shape[1:]).to(dev)
    cat_tensor[0] = encoded_tensor
    for i in tqdm(range(1, tensors.shape[0]), "Encoding tensors"):
        encoded_tensor = taesd.encoder(tensors[i].unsqueeze(0).to(dev))
        cat_tensor[i] = encoded_tensor
    return cat_tensor



if __name__ == "__main__" :
    from saver_loader import ls_with_cache_file_tensor, ls_with_cache_file_tensor_dataset, save_dict_hash
    plt.rcdefaults()
    dataset_loading_parameters = {
        "data_set_name" : "chq",
        "num_samples" : 100,
        "target_labels" : [],
        "image_size" : 512,    
        "normalize" : False,
    }
    device = "cpu" if False else "cuda:0"
    hash = save_dict_hash(dataset_loading_parameters)
    from dataset_loader import load_parquet
    # Load Data
    data_tensor = ls_with_cache_file_tensor_dataset(search_dir="./data/cached_tensors/dataset", 
                                                    data_dir="./data", force_rewrite=False,
                                                    dataset_parameters=dataset_loading_parameters, 
                                                    requirements=[],device=device)
    
    taesd = TAESD(*["ae/taesd/taesd_encoder.pth", "ae/taesd/taesd_decoder.pth"]).to(device)
    encoded_batch_tensor = ls_with_cache_file_tensor(search_dir="./data/cached_tensors/taesd_encoded_dataset", 
                                               preprocess_tensor_function= lambda x : transform_to_latent_space(x,taesd=taesd, dev=device), 
                                               tensor_to_save=data_tensor, device=device, 
                                               requirements=["encoded",hash], force_rewrite=False)
    torch.cuda.empty_cache()

    # Initialize and Fit the Generator
    generator = PCAGMMFaceGenerator(pca_A_dim=256, gmm_components=5, device="cuda:0")
    generator.fit(encoded_batch_tensor)

    seed = 1
    generated_faces = generator.sample(num_samples=1, return_average=False, seed=seed)
    
    # Or generate the overall average face:
    # average_face = generator.sample(num_samples=1, return_average=True)


    single_generated_image = generated_faces[0]

    os.makedirs("./data/cached_tensors/gmm", exist_ok=True)
    torch.save(single_generated_image, "./data/cached_tensors/gmm/celebahq_512_lat2.pt")
    
    for i in range(len(generated_faces)):
        plt.figure(i)
        display_image = torch.clamp(generated_faces[i], 0.0, 1.0).cpu()
        plt.imshow(TF.to_pil_image(display_image))
        plt.axis("off")
    plt.show()