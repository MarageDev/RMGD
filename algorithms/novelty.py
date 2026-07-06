import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import colorsys
import matplotlib.pyplot as plt
def get_color_range(ref_img_nb, end=200):
    # Handle the edge case where 0 or 1 unique patches are found
    if ref_img_nb <= 1:
        return np.array([colorsys.hls_to_rgb(255., 0.5, 0.5)]) # Default to red if only 1 patch type #return np.array([[1.0, 0.0, 0.0]]) 

    clr = np.zeros(dtype=np.float64, shape=(ref_img_nb, 3))
    for i, h in enumerate(np.linspace(0, end, ref_img_nb)): # red to pink color shift
        clr[i, :] = colorsys.hls_to_rgb(h/255., 0.5, 0.5)
    return clr

def mask_img(ref_t: Tensor, comp_t: Tensor):
    # ref_t: (3, H, W), comp_t: (3, H, W)

    diff_map = torch.norm(ref_t - comp_t, p=2, dim=0) # Shape: (H, W)

    # Keep as a boolean mask
    mask = diff_map < 0.1 

    # To multiply a 2D mask with a 3D image, unsqueeze a channel dimension to make it (1, H, W)
    masked_image = mask.unsqueeze(0) * torch.ones_like(ref_t)

    return mask, masked_image


def img_to_tensor(path):
    import torchvision.transforms as transforms
    from PIL import Image
    img = Image.open(path).convert('RGB') # Make RGB
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
    return transform(img)

def tensor_to_numpy_img(tensor:Tensor):
    return np.clip(tensor.permute(1, 2, 0).cpu().detach().numpy(), 0, 1)

@torch.no_grad()
def compare_ref_stack_simple(ref_t: Tensor, comp_t: Tensor):
    
    final = torch.ones_like(ref_t).to('cuda:0') # White canvas (3, H, W)

    clr_rg = torch.tensor(get_color_range(comp_t.shape[0]), dtype=torch.float32).to('cuda:0') # (N, 3)
    
    for i in range(comp_t.shape[0]): 
        mask, _ = mask_img(ref_t, comp_t[i]) # mask shape (H, W)
        mask_3d = mask.unsqueeze(0) # (H, W) to (1, H, W)
  
        color_3d = clr_rg[i].view(3, 1, 1) # (3) to (3, 1, 1)

        final = torch.where(mask_3d, color_3d, final) # Where mask_3d is True apply color_3d. Otherwise keep final.
        
    return final

@torch.no_grad()
def compare_ref_stack(ref_t: torch.Tensor, comp_t: torch.Tensor, threshold: float = 0.1, smooth_kernel: int = 3, distance_gradient=True, spatial_weights: torch.Tensor = None, expand_mosaic_visualisation=True):
    N, C, H, W = comp_t.shape
    device = ref_t.device
    
    # Calculate raw distances across the entire stack
    diff = comp_t - ref_t.unsqueeze(0) 
    raw_distances = torch.norm(diff, p=2, dim=1) # Shape: (N, H, W)
    
    # Calculate smoothed distances for spatial voting
    if spatial_weights is not None:
        pad = spatial_weights.shape[0] // 2 # spatial_weights (patchsize, patchsize)
        kernel = spatial_weights.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, patchsize, patchsize)
        
        diff = torch.sum(diff ** 2, dim=1, keepdim=True)
        smoothed_distances = F.conv2d(diff, kernel, padding=pad).squeeze(1)

    elif smooth_kernel > 1:
        # if no spatial_weights, use standard uniform average pooling
        pad = smooth_kernel // 2
        smoothed_distances = F.avg_pool2d(
            raw_distances.unsqueeze(1), kernel_size=smooth_kernel, stride=1, padding=pad
        ).squeeze(1)
    
    else:
        smoothed_distances = raw_distances

    # Get the choices from both methods
    _, best_indices_smooth = torch.min(smoothed_distances, dim=0) # (H, W)
    min_raw_distances, best_indices_raw = torch.min(raw_distances, dim=0) # (H, W)
    
    #  Check if the smoothed choice passes the threshold
    chosen_raw_smooth = raw_distances.gather(0, best_indices_smooth.unsqueeze(0)).squeeze(0)
    mask_smooth = chosen_raw_smooth < threshold
    
    #  Fallback : Use smooth choice if valid, otherwise fall back to raw best choice
    mask_raw = min_raw_distances < threshold
    fallback_mask = mask_raw & ~mask_smooth # raw and not smooth
    
    final_indices = torch.where(fallback_mask, best_indices_raw, best_indices_smooth)
    final_mask = mask_smooth | mask_raw # Valid if either passes, smooth or raw
    
    # Get the final chosen distances for each pixel
    final_distances = raw_distances.gather(0, final_indices.unsqueeze(0)).squeeze(0)
    
    ##################################
    # Retrieve the predominant image
    ##################################
    counts:tuple[torch.Tensor,torch.Tensor] = final_indices.unique(return_counts=True)
    predominant_tensor_idx = counts[0][counts[1].argmax()]
    
    
    
    ####################################
    # Color map of regions
    ######################################
    
    # Map final indices to colors
    final = torch.ones_like(ref_t) # White canvas
    
    ### Color range calibration (ot get higher contrast of colors with only a separation of the spectrum based  on the number of patches)

    mask_pathces_nb = final_indices.unique().shape[0]# retrieve the number of mask patches
    
    clr_rg = torch.tensor(get_color_range(mask_pathces_nb), dtype=torch.float32, device=device)

    # Create a mapping from original indices to color indices (0 to patch number)
    unique_indices = final_indices.unique()
    index_mapping = torch.full((N,1), -1, dtype=torch.long, device=device)
    for new_idx, orig_idx in enumerate(unique_indices):
        index_mapping[orig_idx] = new_idx

    # Remap final_indices using the mapping
    remapped_indices = index_mapping[final_indices.flatten()]

    # Retrieve he color
    colors = clr_rg[remapped_indices]
    color = colors.view(H, W, 3).permute(2, 0, 1)
    
    if distance_gradient:
        normalized_distances = torch.clamp(final_distances / threshold, 0, 1)
        distance_factor = 1.0 - normalized_distances
        color = color * distance_factor.unsqueeze(0)
    
    
    final = torch.where(final_mask.unsqueeze(0), color, final)
    
    #############################
    # Image reference map/mosaic of regions
    # Display the correspoding pixel of the reference image in the reference tensor where it is supposedly taken from based on the mask view
    ###########################################
    
    mosaic_final = torch.ones_like(ref_t) # White canvas
    for i in unique_indices:
        #mosaic_final = torch.where(final_indices == i, comp_t[i], mosaic_final)
        pixel_mask = (final_indices == i) & (final_mask if expand_mosaic_visualisation else True)
        mosaic_final = torch.where(pixel_mask.unsqueeze(0), comp_t[i], mosaic_final)
    
    ################################
    # Display the areas where the predominant image is
    ###################################################
    
    mosaic_predom = torch.ones_like(ref_t) # White canvas
    pixel_mask = (final_indices == predominant_tensor_idx) & (final_mask if expand_mosaic_visualisation else True)
    mosaic_predom = torch.where(pixel_mask.unsqueeze(0), comp_t[predominant_tensor_idx], mosaic_predom)
    
    
    ################################
    # Display the areas where the predominant centor of interest(coi) image is (searching in a centered rect to search for faces)
    # + mosaic of coi
    ###################################################
    
    w_ratio = 0.5
    h_ratio = 0.7
    #0.25 0.75
    h_start, h_end = int(H * (1.-h_ratio)/2.), int(H * ((h_ratio + 1.)/2.))
    w_start, w_end = int(W * (1.-w_ratio)/2.), int(W * ((w_ratio + 1.) /2.))
    
    
    # Slice the final indices to only look at that central rect
    sliced_final_indices = final_indices[h_start:h_end, w_start:w_end]

    counts: tuple[torch.Tensor, torch.Tensor] = sliced_final_indices.unique(return_counts=True)
    predominant_coi_tensor_idx = counts[0][counts[1].argmax()]
    
    mosaic_coi_predom = torch.ones_like(ref_t) # White canvas
    pixel_mask_coi = (final_indices == predominant_coi_tensor_idx) & (final_mask if expand_mosaic_visualisation else True)
    mosaic_coi_predom = torch.where(pixel_mask_coi.unsqueeze(0), comp_t[predominant_coi_tensor_idx], mosaic_coi_predom)
    
    predom_coi_image  = comp_t[predominant_coi_tensor_idx]
    brightness = 0.6
    predom_coi_image[:, h_start:h_end, w_start:w_end] = predom_coi_image[:, h_start:h_end, w_start:w_end]*brightness + (1.- brightness)
    
    return final, mosaic_final, comp_t[predominant_tensor_idx], mosaic_predom, predom_coi_image, mosaic_coi_predom, normalized_distances.unsqueeze(0)
    #       0       1           2                               3               4                       5               6

if __name__ == "__main__":
    data_tensor = torch.load("./data/cached_tensors/dataset/tensor_dataset_4f4934b05986be2050108a794ee35105e817d299f3981f0327a61b052bd6c589.pt", map_location='cuda:0')

    fig = plt.figure(layout="constrained")
    layout = """
    SSCPI
    SSMDO
    """
    
    axs = fig.subplot_mosaic(mosaic=layout)
    ref_tensor = img_to_tensor("novelty_test.jpg").to('cuda:0')
    comp = compare_ref_stack(ref_tensor,data_tensor,
                             smooth_kernel=5,threshold=0.2, 
                             distance_gradient=True, spatial_weights=None)
    
    axs["S"].imshow(tensor_to_numpy_img((ref_tensor +1)/2)) # synthesis
    
    axs["C"].imshow(tensor_to_numpy_img(comp[0])) # patches
    
    axs["M"].imshow(tensor_to_numpy_img(comp[1])) # mosaic of ref images
    axs["P"].imshow(tensor_to_numpy_img(comp[2])) # predominant image
    axs["D"].imshow(tensor_to_numpy_img(comp[3])) # predom mosaic
    axs["I"].imshow(tensor_to_numpy_img(comp[4])) # predom coi image
    axs["O"].imshow(tensor_to_numpy_img(comp[5])) # predom coi mosaic
    plt.show()