import os
import torch

def split_tensor_file(input_file_path: str, num_splits: int, output_dir: str = "./splits") -> list:
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original tensor
    print(f"Loading {input_file_path}...")
    tensor = torch.load(input_file_path)

    B, C, H, W = tensor.shape
    if num_splits > B:
        raise ValueError(f"Cannot split into {num_splits} files; batch size is only {B}.")
        
    print(f"Original tensor shape: {tensor.shape}")
    
    # Calculate chunk sizes
    base_chunk_size = B // num_splits
    remainder = B % num_splits
    
    split_sizes = [base_chunk_size + (1 if i < remainder else 0) for i in range(num_splits)]
    
    # Split the tensor along the batch dimension
    tensor_splits = torch.split(tensor, split_sizes, dim=0)
    
    saved_files = []
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    
    for i, split_tensor in enumerate(tensor_splits):
        output_filename = f"{base_name}_split_{i+1:02d}.pt"
        output_path = os.path.join(output_dir, output_filename)
        
        cloned_split = split_tensor.clone()
        
        torch.save(cloned_split, output_path)
        saved_files.append(output_path)
        print(f"Saved: {output_path} (Shape: {list(cloned_split.shape)})")
        
    print(f"\nSuccessfully split tensor into {num_splits} files.")
    return saved_files


if __name__ == "__main__":
    split_files = split_tensor_file(r"C:\Users\Mahe\Development\GitHub\RMGD\data\cached_tensors\taesd_encoded_dataset\fully_encoded_celebahq_dataset.pt", num_splits=3, output_dir="./my_splits")