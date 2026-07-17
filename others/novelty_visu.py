from algorithms.novelty import *
import matplotlib.pyplot as plt
import numpy as np
import torch

image_on_line = 2
number_of_images = image_on_line * 5

fig, axes = plt.subplots(5, image_on_line * 2, figsize=(image_on_line * 5, 8))

data_tensor = torch.load(
    "data/saved_tensors/tensor_cache_0_aa4db64a2e767a0492224b35efe584452423cee26e90a14789f94db9f646d05b.pt", 
    map_location='cuda:0'
)

for i in range(number_of_images):
    
    # Calculate grid position
    row = i // image_on_line             
    col_base = (i % image_on_line) * 2   
    
    ax_ref = axes[row, col_base]
    ax_mask = axes[row, col_base + 1]
    
    # Process tensors
    ref_tensor = img_to_tensor(f"ref_img{i}.png").to('cuda:0')
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

plt.tight_layout()
plt.show()