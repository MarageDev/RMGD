from algorithms.algo1 import algo1_diffusion_inverse
from algorithms.algo2 import algo2_kwatra_original
from algorithms.algo3 import algo3_kwatra_modifie
from algorithms.algo4 import algo4_nifty
from algorithms.algo5 import algo5

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import perf_counter

from data_tensor_loader import load_data_to_tensor
ttot = perf_counter()
device = torch.device('cpu')#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##### MAIN PARAMETERS
""" 
Availible datasets :
celeba  (c)         cifar   (ci)            cat     (ca)
fashion (f)         mnist   (m)             afhq    (af)
art     (a)         obama   (o)             flickr  (fl)
dog     (d)         grumpy  (g)             panda   (p)
"""
force_reload_tensor = False
dataset_loading_parameters = {
    "data_set_name" : "c",
    "num_samples" : 500,
    "target_labels" : [4],
    "image_size" : 32,    
}

patch_size = 9
iteration = 10

# Sub parameters (no specific need to change them by default)
data_dir = "./data"
image_on_line = 2
mask_type = "gaussian"
manual_seed = None



torch.cuda.empty_cache()
data_tensor = load_data_to_tensor(None, data_dir=data_dir, dataset_loading_parameters=dataset_loading_parameters, force_reload_tensor=force_reload_tensor)
data_tensor = data_tensor.to(device)

def tensor_to_img(tensor):
    img = np.clip((tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2, 0, 1)
    # If the last dimension is 1, drop it
    if img.shape[-1] == 1:
        img = img.squeeze(-1) 
        
    return img

## Generation
algos = [
    algo1_diffusion_inverse,
    algo2_kwatra_original,
    algo3_kwatra_modifie,
    algo4_nifty,
    algo5
]




lines = len(algos)
number_of_images = image_on_line * lines
fig,axes = plt.subplots(lines, image_on_line, figsize=(image_on_line, lines))
axes_flat = np.ndarray.flatten(axes)

all_sequences = []
seeds = [manual_seed + i if manual_seed is not None else torch.randint(0,9999999,(1,1)) for i in range(image_on_line)]

if data_tensor.shape[1] == 1 : plt.set_cmap("gray")
# Algo

for i in range(number_of_images):
    
    # Position
    row = i // image_on_line   
    col = (i % image_on_line)
    
    s = perf_counter()
    if row != 4 : results = algos[row](data_tensor, patchsize=patch_size, N=iteration, device=device,seed=seeds[col])
    else : results = algos[row](data_tensor, patchsize=patch_size, N=iteration, device=device,mask_weight_type =mask_type,seed=seeds[col])
    last_result = results[-1]
    print(f"Execution time: {perf_counter() - s:.5f}s")
    all_sequences.append(results)

    #plt.imsave(f"ref_img{i}.png",tensor_to_img(last_result))

    axes_flat[i].imshow(tensor_to_img(last_result))

[ax.axis('off') for ax in axes_flat]

ims = []
num_steps = min(len(seq) for seq in all_sequences) # because sometimes algo 2 and 3 stops sooner (loop break if there's no change, line 7)
for step in range(num_steps):
    frame_artists = []
    for i in range(number_of_images):
        img_data = tensor_to_img(all_sequences[i][step])
        im = axes_flat[i].imshow(img_data, animated=True)
        frame_artists.append(im)
    
    ims.append(frame_artists)
print(perf_counter()-ttot)
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=2000)


# Novelty map visualisation

from algorithms.novelty import *

image_on_line = 2
number_of_images = image_on_line * lines

fig, axes = plt.subplots(lines, image_on_line * 2, figsize=(image_on_line * lines, 8), squeeze=False)

for i in range(number_of_images):
    
    # Calculate grid position
    row = i // image_on_line             
    col_base = (i % image_on_line) * 2   
    
    ax_ref = axes[row, col_base]
    ax_mask = axes[row, col_base + 1]
    
    # Process tensors
    ref_tensor = img_to_tensor(f"ref_img{i}.png").to(device)
    comparison = compare_ref_stack(ref_tensor, data_tensor, smooth_kernel=3)
    
    # Display Reference Image
    ref_img_np = tensor_to_numpy_img((ref_tensor + 1) / 2) 
    ax_ref.imshow(ref_img_np)
    ax_ref.set_title(f"Algo {i//2 + 1} Output ({i})")
    ax_ref.axis('off')

    # Display Masked Image
    ax_mask.imshow(tensor_to_numpy_img(comparison))
    ax_mask.set_title(f"Mask {i}")
    ax_mask.axis('off')


plt.show()