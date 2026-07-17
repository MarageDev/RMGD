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
        print("Data flattened")
        
        self.mean = flattened.mean(dim=0, keepdim=True).to(self.device)
        
        # PCA to pca_A_dim dimensions
        U, S, self.V = torch.pca_lowrank(flattened - self.mean, q=self.pca_A_dim, center=True, niter=self.pca_iter) 
        pca_features = (flattened - self.mean) @ self.V # project
        print("Model PCA done")
        
        # Fit GMM on PCA
        self.gmm = GaussianMixtureGPU(
            n_components=self.gmm_components, 
            n_features=self.pca_A_dim, 
            max_iter=100, # TODO 1000
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
    
    def sample_pc(self, pc_index:int = 0, value:float = 0., return_average=False):
        """
        Generates a sequence of samples by sweeping a single Principal Component
        from a minimum to a maximum value, keeping all other components at the mean.
        """
        if self.V is None:
            raise RuntimeError("Model is not fitted yet, call .fit() first")

        if return_average:
            pca_vectors = torch.zeros((1, self.pca_A_dim), device=self.device, dtype=torch.float32)
        else:
            # Sample a latent vector from the fitted GMM
            gmm_samples = self.gmm.sample(1)[0]
            pca_vectors = torch.as_tensor(gmm_samples, device=self.device, dtype=torch.float32)
            
            if pca_vectors.ndim == 1: # batch dim in case it's of dim 1
                pca_vectors = pca_vectors.unsqueeze(0)

        pca_vectors[:, pc_index] = value # set the pc inedx to the value

        reconstructed = (pca_vectors @ self.V.t()) + self.mean
        
        return reconstructed.view(1, *self.image_shape)

    def sample_multiple_pcs(self,pc_weights: dict[int, float] = None, return_average: bool = False):
        """
        Generate a sample by perturbing multiple principal components simultaneously
        """
        if self.V is None:
            raise RuntimeError("Model is not fitted yet, call .fit() first")

        if pc_weights is None:
            pc_weights = {}

        pc_items = list(pc_weights.items())

        if return_average:
            pca_vectors = torch.zeros((1, self.pca_A_dim), device=self.device, dtype=torch.float32)
        
        else:
            gmm_samples = self.gmm.sample(1)[0]
            pca_vectors = torch.as_tensor(gmm_samples, device=self.device, dtype=torch.float32)
            
            if pca_vectors.ndim == 1:# add a batch dim incase it's of dim 1
                pca_vectors = pca_vectors.unsqueeze(0)

        for pc_index, value in pc_items:
            pca_vectors[:, int(pc_index)] = float(value)

        reconstructed = (pca_vectors @ self.V.t()) + self.mean
        return reconstructed.view(1, *self.image_shape)


    def sample_along_pc(self, pc_index: int = 0, num_steps: int = 5, sweep_range: tuple = (-15.0, 15.0)):
        """
        Generates a sequence of samples by sweeping a single Principal Component
        from a minimum to a maximum value, keeping all other components at the mean (0).
        """
        if self.V is None:
            raise RuntimeError("Model is not fitted yet, call .fit() first")

        # Create a batch of base vectors in PCA space
        pca_vectors = torch.zeros((num_steps, self.pca_A_dim), device=self.device, dtype=torch.float32)

        # Create the sequence of values for the target feature
        sweep_values = torch.linspace(sweep_range[0], sweep_range[1], steps=num_steps, device=self.device)

        # Add the sweep values at the Principal Component index (only not 0 at pc_index)
        pca_vectors[:, pc_index] = sweep_values

        # Inverse PCA to project back to the latent/image space
        reconstructed = (pca_vectors @ self.V.t()) + self.mean
        
        return reconstructed.view(num_steps, *self.image_shape)

    def get_pc_distribution(self, encoded_batch_tensor: torch.Tensor, component_idx: int = 0) -> tuple[float, float, float]:
        """
        Get data distribution for a specific PCA component : min, max, mean
        """
        if self.gmm is None or self.V is None:
            raise RuntimeError("Model is not fitted yet, call .fit() first")

        # Project the input data into the PCA space
        flattened = encoded_batch_tensor.view(encoded_batch_tensor.shape[0], -1)
        pca_features = (flattened - self.mean) @ self.V

        # Extract the specific 1D component
        pca_1d = pca_features[:, component_idx].cpu().numpy()

        return float(pca_1d.min()), float(pca_1d.max()), float(pca_1d.mean())
    
    # Debug plots
    def display_centroids_images(self, decoder=None, grid_cols=5):
        if self.gmm is None or self.V is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

        # Extract centroids from the GMM
        centroids_pca = self.gmm.means_
        centroids_pca = torch.as_tensor(centroids_pca, device=self.device, dtype=torch.float32)

        # Inverse PCA to project back to the original space
        reconstructed_centroids = (centroids_pca @ self.V.t()) + self.mean
        reconstructed_centroids = reconstructed_centroids.view(self.gmm_components, *self.image_shape)
        
        # Decode latents to RGB if a decoder is passed, else don't do anything
        if decoder is not None:
            with torch.no_grad():
                reconstructed_centroids = decoder(reconstructed_centroids)
                
        # Plotting the Grid
        num_centroids = self.gmm_components
        grid_rows = (num_centroids + grid_cols - 1) // grid_cols
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
        
        axes = [axes] if num_centroids == 1 else axes.flatten() # flattened axes + quick fix for single case
        
        for i in range(len(axes)):
            if i < num_centroids: # in case there's less centroids
                display_image = torch.clamp(reconstructed_centroids[i], 0.0, 1.0).cpu()
                axes[i].imshow(TF.to_pil_image(display_image))
                axes[i].set_title(f"Centroid {i}")
            axes[i].axis("off")
            
        plt.tight_layout()
        plt.show()
    
    def display_2d_distribution_map(self, encoded_batch_tensor: torch.Tensor, components_observed_2d:tuple[int,int]=(0,2)):
        """
        Plot a 2D scatter map of the data distribution and the GMM centroids
        """
        if self.gmm is None or self.V is None:
            raise RuntimeError("Model is not fitted yet,call .fit() first")

        # Project the input data into the PCA space
        flattened = encoded_batch_tensor.view(encoded_batch_tensor.shape[0], -1)
        pca_features = (flattened - self.mean) @ self.V

        components = [components_observed_2d[0],components_observed_2d[1]+1]
        
        # Extract the 2 observed components (for x & y plot)
        pca_2d = pca_features[:, components[0]:components[1]].cpu().numpy()

        # Get cluster assignments to color the distribution (like https://scikit-learn.org/stable/modules/mixture.html#:~:text=2.1.1.-,Gaussian,-Mixture%23 https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.predict
        with torch.no_grad():
            labels = self.gmm.predict(pca_features).cpu().numpy()
        
        # Extract centroids and get the 2 obeserved components
        centroids_pca = self.gmm.means_
            
        centroids_2d = centroids_pca[:, components[0]:components[1]].cpu().numpy()

        # Plot the 2D Map
        plt.figure(figsize=(10, 8))
        
        # Plot the distribution
        scatter = plt.scatter(
            pca_2d[:, 0], pca_2d[:, 1], 
            c=labels, cmap='viridis', s=15, alpha=0.6, label='Data Points'
        )
        
        # Plot the GMM Centroids
        plt.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1], 
            c='tab:red', marker='x', s=250, linewidths=1.5, label='GMM Centroids'
        )

        plt.title('2D Map of Latent Space (2 components)')
        plt.xlabel(f'Principal Component {components[0]}')
        plt.ylabel(f'Principal Component {components[1]}')
        plt.legend()
        
        # Only add colorbar if there are cluster labels
        if len(np.unique(labels)) > 1:
            plt.colorbar(scatter, label='GMM Cluster Assignment')
            
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
    def display_1d_distribution(self, encoded_batch_tensor: torch.Tensor, component_idx: int = 0) -> tuple[float, float, float]:
        """
        Plot a 1D histogram of the data distribution for a specific PCA component, and return the min, max, and mean values.
        
        min, max, mean
        """
        if self.gmm is None or self.V is None:
            raise RuntimeError("Model is not fitted yet, call .fit() first")

        # Project the input data into the PCA space
        flattened = encoded_batch_tensor.view(encoded_batch_tensor.shape[0], -1)
        pca_features = (flattened - self.mean) @ self.V

        # Extract the specific 1D component
        pca_1d = pca_features[:, component_idx].cpu().numpy()

        # Calculate statistics
        min_val = float(pca_1d.min())
        max_val = float(pca_1d.max())
        mean_val = float(pca_1d.mean())

        # Plot the 1D Distribution (Histogram)
        plt.figure(figsize=(5, 4))
        
        # Plot histogram
        plt.hist(pca_1d, bins=50, color='tab:orange', alpha=0.8, density=True)
        
        # Add vertical lines to see stats
        plt.axvline(mean_val, color='tab:red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
        plt.axvline(min_val, color='tab:green', linestyle='dashed', linewidth=1, label=f'Min: {min_val:.2f}')
        plt.axvline(max_val, color='tab:blue', linestyle='dashed', linewidth=1, label=f'Max: {max_val:.2f}')

        plt.title(f'1D Distribution of Principal Component {component_idx}')
        plt.xlabel(f'Value of Principal Component {component_idx}')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='-', alpha=0.25)
        plt.tight_layout()
        plt.show()

        return min_val, max_val, mean_val
    
    # Saving and loading
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
        generator = PCAGMMFaceGenerator.load(f"{search_dir}/{existant_file_path}", device=device)
        generator.seed = seed
        print(f"Loaded PCAGMM model from {search_dir}/{existant_file_path}")
        
        return generator
    else:
        generator = PCAGMMFaceGenerator(pca_A_dim=settings["PCA_dim"], gmm_components=settings["GMM_comp"], pca_iter=settings["PCA_ITER"],gmm_init_iter=["GMM_INIT_ITER"],device=device)
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

def traversal(generator:PCAGMMFaceGenerator,pca_component:int,steps:int,sweep_range:tuple[float,float]=(-100.,100.), taesd:TAESD=None, plot=False):
    # Generate X steps along that feature
    traversal_latents = generator.sample_along_pc(
        pc_index=pca_component, 
        num_steps=steps, 
        sweep_range=sweep_range
    )

    # Decode the latents to RGB images
    with torch.no_grad():
        traversal_images = taesd.decoder(traversal_latents)
        
    # Plot them side-by-side
    if plot :
        fig, axes = plt.subplots(1, steps, figsize=(15, 3))
        for i in range(steps):
            display_image = torch.clamp(traversal_images[i], 0.0, 1.0).cpu()
            axes[i].imshow(TF.to_pil_image(display_image))
            axes[i].axis("off")
            
        plt.suptitle(f'Moving Principal Component {pca_component} [{ round(sweep_range[0],2)}:{round(sweep_range[1],2)}]')
        plt.tight_layout()
        plt.show()  

    return traversal_latents
"""if __name__ == "__main__" :
    from saver_loader import ls_with_cache_file_tensor, ls_with_cache_file_tensor_dataset, save_dict_hash
    plt.rcdefaults()
    dataset_loading_parameters = {
        "data_set_name" : "chq",
        "num_samples" : 500,
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
    generator = PCAGMMFaceGenerator(pca_A_dim=128, gmm_components=5, pca_iter=15,gmm_init_iter=2,device="cuda:0")
    generator.fit(encoded_batch_tensor)
"""

if __name__ == "__main__":
    from saver_loader import ls_with_cache_file_tensor, ls_with_cache_file_tensor_dataset, save_dict_hash
    from time import perf_counter
    plt.rcdefaults()
    dataset_loading_parameters = {
        "data_set_name" : "chq_jpg",
        "num_samples" : None,
        "target_labels" : [],
        "image_size" : 512,    
        "normalize" : False,
    }
    device = "cpu" if False else "cuda:0"
    hash = save_dict_hash(dataset_loading_parameters)
    taesd = TAESD(*["ae/taesd/taesd_encoder.pth", "ae/taesd/taesd_decoder.pth"]).to(device)
    t0 = perf_counter()
    encoded_batch_tensor = torch.load("data/cached_tensors/taesd_encoded_dataset/fully_encoded_celebahq_dataset.pt", weights_only=False, map_location=device)
    torch.cuda.empty_cache()   
    print(perf_counter() - t0)
    
    PCAGMM_SETTINGS = {
    "PCA_dim": 256,
    "GMM_comp": 1,
    "PCA_ITER":50, 
    "GMM_INIT_ITER":1
    }
    t0 = perf_counter()
    pcagmm_generator = quick_create_model([hash, encoded_batch_tensor], PCAGMM_SETTINGS, seed=None, force_rewrite=False)#.sample(1, return_average=False, seed = None)
    #pcagmm_generator.display_2d_distribution_map(encoded_batch_tensor,(0,1))
    print(perf_counter() - t0)

    x = pcagmm_generator.sample_multiple_pcs(pc_weights={5 : -60,3 : 82, 8:-63, 18:38, 39:-18, 7:-59, 0:80},return_average=False)
    print(x.shape)
    plt.imshow(TF.to_pil_image(torch.clamp(taesd.decoder(x)[0],0.,1.)))
    plt.show()
    
    
    # for x in range(200,257):
    #     nb = x
    #     min, max, _ = pcagmm_generator.get_pc_distribution(encoded_batch_tensor, nb)
    #     traversal(pcagmm_generator, nb, 7, (min,max), taesd=taesd, plot=True)
    