from dataclasses import dataclass
from torch import Tensor

@dataclass
class AppState:
    iterations: int = 5
    patch_size_pow : int = 3
    patch_size: int = 2 ** 3
    stride: int = 3
    init_noisefact: float = 0.5
    algo_samples: int = 500
    seed: int = 0
    use_seed: bool = False
    weight_type: str = "default"
    scheduler: str = "linear"
    device: str = "cuda:0"
    
    sampled_tensor:Tensor = None # Retrieved from the CPAGMM face generator
    init_tensor:Tensor = None # sampled_tensor + noise blend for algorithm initialisation
    result_tensor:Tensor = None # final result of the generation
    
    batch_tensor:Tensor = None # Batch of images used to run the algorithm clean patch retrieval with
    batch_tensor_path : str = "./data/cached_tensors/taesd_encoded_dataset/fully_encoded_celebahq_dataset_split_01.pt"
    pcagmm_model_path: str = "./data/cached_tensors/pcagmm/tensor_pcagm_512_10.pt" # 512 pca, 10 gmm