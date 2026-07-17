import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.state_type import AppState
import numpy as np
import torch
from torch import Tensor

import gradio as gr

# Algorithms to import
from algorithms.algorithm_PCAGMM_latent import ALGO_LATENT
from algorithms.pcagmm_generator import PCAGMM_GENERATOR
from algorithms.novelty import compare_ref_stack
from algorithms.taesd.taesd import TAESD

# Helpers
def tensor_to_numpy_img(tensor:Tensor):
    """
    3 channel input tensor (C H W) to numpy image (H W C)
    """
    if tensor.dim() == 4: # Handle batche dim
        tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]
    img = np.clip(tensor.permute(1, 2, 0).cpu().detach().numpy(), 0, 1) # Handle 3C
    if img.shape[-1] == 1: # Handle 1C
        img = img.squeeze(-1)
    
    return img

def decode_tensor(tensor:Tensor, device):
    taesd = TAESD(*["algorithms/taesd/taesd_encoder.pth", "algorithms/taesd/taesd_decoder.pth"]).to(device)
    return taesd.decoder(tensor.to(device)).clamp(0, 1)

# Computation Functions
def process_run(state:AppState):
    _set_seed(state)
    
    if state.init_tensor == None : 
        gr.Warning("""No initialisation image, press "Resample Initialisation" if you wish to define one. Will use random generation instead for now.""")
        process_init(state)
        process_algo_init(state)
    
    encoded_tensor = torch.load(state.batch_tensor_path, weights_only=False, map_location=state.init_tensor.device)[0:state.algo_samples]
    results_lat = ALGO_LATENT(encoded_tensor, state.init_tensor, state.patch_size, state.iterations, state.scheduler, "cpu", state.weight_type, seed=None if state.use_seed != True else state.seed, stride=state.stride)
    
    for im in results_lat : 
        final_result_rgb = tensor_to_numpy_img(decode_tensor(im,state.device)[0])
        # Save to state and move to CPU to avoid ZeroGPU memory boundary issues
        state.result_tensor = im.cpu() 
        yield final_result_rgb, state

def process_init(state:AppState):
    _set_seed(state)
    
    pcagmm_generator = PCAGMM_GENERATOR.load(
        #"./cached_tensors/pcagmm/1.pt", # 256 pca, 1000 gmm
        state.pcagmm_model_path, # 512 pca, 10 gmm
        device="cpu")
    
    sample = pcagmm_generator.sample(1, return_average=False, seed = state.seed) # B C H W
    
    state.sampled_tensor = sample.cpu()
    
    converted = decode_tensor(sample,state.device) # -> B C H W
    return tensor_to_numpy_img(converted[0]), state # H W C

def process_algo_init(state:AppState):
    _set_seed(state)
    
    x_noise = state.sampled_tensor.to(state.device) # Ensure it's on device
    device = x_noise.device
    x_noise = (1. - state.init_noisefact ) * x_noise + state.init_noisefact * torch.randn_like(x_noise, device=device)
    
    state.init_tensor = x_noise.cpu()
    return tensor_to_numpy_img(decode_tensor(x_noise, state.device)[0]), state


def process_advanced_display(state:AppState):
    # [img_patch_regions, img_novelty_map, img_predominant_mosaic, img_predominant, img_coi_mosaic, img_coi]
    encoded_tensor = torch.load(state.batch_tensor_path, weights_only=False, map_location=state.device)[0:state.algo_samples]

    comparison = compare_ref_stack(
        state.result_tensor[0].to(state.device), 
        encoded_tensor.to(state.device),
        threshold=0.9, # Difference > 90% displayed as white pixels
        distance_gradient=True,
        spatial_weights=None,
        smooth_kernel=state.patch_size + ( 1 if state.patch_size % 2 == 0 else 0 )
    )
    print(state.patch_size + ( 1 if state.patch_size % 2 == 0 else 0 ))
    return [
        tensor_to_numpy_img(comparison[0]), # patch region
        tensor_to_numpy_img(comparison[6]), # novelty
        tensor_to_numpy_img(decode_tensor(comparison[3],state.device)), # predominant mosaic
        tensor_to_numpy_img(decode_tensor(comparison[2],state.device)), # predominant image
        tensor_to_numpy_img(decode_tensor(comparison[5],state.device)), # coi mosaic
        tensor_to_numpy_img(decode_tensor(comparison[4],state.device)),# coi image
    ]
    # tensor_to_numpy_img(decode_tensor(comparison[1],state.device))
    
def _set_seed(state:AppState):
    if state.use_seed:
        torch.manual_seed(state.seed)
        torch.cuda.manual_seed_all(state.seed)
        np.random.seed(state.seed) # For numpy stuff, set the same seed as torch
        print(f"Seed set to {state.seed}")

def update_algorithm_dataset(state:AppState):
    if state.batch_tensor_path == "" or state.batch_tensor_path is None : 
        return gr.update(value=1,maximum=2,interactive = False)
    state.batch_tensor = torch.load(state.batch_tensor_path, weights_only=False)
    return gr.update(maximum=state.batch_tensor.shape[0], value=state.batch_tensor.shape[0], interactive = True)