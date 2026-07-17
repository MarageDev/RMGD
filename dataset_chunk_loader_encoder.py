from others.saver_loader import has_corresponding_reqs, save_dict_hash
from time import perf_counter
import torch
from algorithms.taesd.taesd import TAESD
from tqdm import tqdm
import os
import gc
from torchvision.io import decode_image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# PARAMETERS

device = "cuda:0" if torch.cuda.is_available() else "cpu"
root = './data'
image_source = "./data/celebahq_jpg"
image_size = 512
batch_size = 1
num_workers = 12

####################################################################

class ImageDataset(Dataset):
    """Custom PyTorch Dataset to handle loading and resizing in parallel"""
    def __init__(self, image_dir, size):
        self.image_dir = image_dir
        self.size = size
        self.files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.image_dir, self.files[idx])
        try:
            img_tensor = decode_image(file_path, mode="RGB")
            img_tensor = TF.resize(img_tensor, [self.size, self.size], antialias=True)
            # Normalize and clap
            img_tensor = torch.clamp(img_tensor.float() / 255., 0., 1.)
            return img_tensor
        except Exception as e:
            print(f"\nError loading {self.files[idx]}: {e}")
            # Return a blank tensor to avoid crash 
            return torch.zeros((3, self.size, self.size), dtype=torch.float32)

####################################################################

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # Load model
    taesd = TAESD("ae/taesd/taesd_encoder.pth", "ae/taesd/taesd_decoder.pth").to(device)
    taesd.eval() # slightly faster with that
    
    # Setup Dataset and DataLoader
    dataset = ImageDataset(image_source, image_size)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    encoded_chunks = []
    
    t0 = perf_counter()
    print(f"Starting encoding of {len(dataset)} images...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches"):
            batch = batch.to(device, non_blocking=True) # Move batch to GPU

            encoded_batch = taesd.encoder(batch)
            
            encoded_chunks.append(encoded_batch.cpu()) # Move back to CPU and store

            del batch, encoded_batch # Clean up leftover tensors (for VRAM)
    
    # Combine everything at the end on CPU
    print("Concatenating final tensor...")
    final_encoded_tensor = torch.cat(encoded_chunks, dim=0)
    
    # Final memory delete
    del encoded_chunks
    gc.collect()
    torch.cuda.empty_cache()
        
    print(f"Total computation : {perf_counter() - t0:.2f} seconds")
    
    # Save tensor to fisk
    save_dir = os.path.join(root, "cached_tensors", "taesd_encoded_dataset")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = "fully_encoded_celebahq_dataset_ttttt"
    save_path = os.path.join(save_dir, f"{filename}.pt")
    
    torch.save(final_encoded_tensor, save_path)
    print(f"Saved successfully to {save_path} with shape {final_encoded_tensor.shape}")