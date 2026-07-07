import torch
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10, CelebA, FashionMNIST, Flickr8k
import torchvision.transforms as transforms

def load_mnist(root='./data', train=True, num_samples=None, target_labels=None, image_size=28, **kwargs):
    if image_size is not None : 
        transform = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    else : 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    
    dataset = MNIST(root=root, train=train, download=True, transform=transform)
    
    images = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        
        if target_labels is None or len(target_labels) == 0:
            condition = True
        else:
            condition = label in target_labels
        if condition:
            images.append(img)
            if num_samples and len(images) >= num_samples:
                break

    return torch.stack(images) if len(images) > 0 else torch.empty(0)

def load_mnist_fashion(root='./data', train=True, num_samples=None, target_labels=None, image_size=28, **kwargs):
    if image_size is not None : 
        transform = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    else : 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
    dataset = FashionMNIST(root=root, train=train, download=True, transform=transform)
    
    images = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        
        if target_labels is None or len(target_labels) == 0:
            condition = True
        else:
            condition = label in target_labels
        if condition:
            images.append(img)
            if num_samples and len(images) >= num_samples:
                break

    return torch.stack(images) if len(images) > 0 else torch.empty(0)

def load_cifar10(root='./data', train=True, num_samples=None, target_labels=None, image_size=32, **kwargs):
    if image_size is not None : 
        transform = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else : 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = CIFAR10(root=root, train=train, download=True, transform=transform)
    images = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if target_labels is None or len(target_labels) == 0:
            condition = True
        else:
            condition = label in target_labels
        if condition:
            images.append(img)
            if num_samples and len(images) >= num_samples:
                break
    return torch.stack(images)if len(images) > 0 else torch.empty(0)

class LooseCelebA(CelebA): # Force check integrity to true because pytorch keeps saying it's false
    def _check_integrity(self) -> bool:
        return True

def load_celeba(root='./data', split='train', num_samples=None, target_labels=None, match_any=False, image_size=64, normalize=True,**kwargs):
    from tqdm import tqdm
    if image_size is not None : 
        transform = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    else : 
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    if normalize : transform = transforms.Compose([*transform.transforms, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = LooseCelebA(root=root, split=split, download=False, transform=transform)
    
    images = []
    for i in tqdm(range(len(dataset)), "Searching through celeba"):
        img, attrs = dataset[i]
        
        # If no attributes are specified, get everything
        if target_labels is None or len(target_labels) == 0:
            condition = True
        else:
            # Search for the attributes
            if match_any:
                condition = any(attrs[idx] == 1 for idx in target_labels)
            else:
                condition = all(attrs[idx] == 1 for idx in target_labels)
                
        if condition:
            images.append(img)
            if num_samples and len(images) >= num_samples:
                break

    return torch.stack(images)


def load_jpg_folder(root='./data', num_samples=None, image_size=None, **kwargs):
    from PIL import Image
    import os
    """
    Loads JPG images from a directory, sorts them alphabetically, 
    and grabs the first `num_samples`.
    Returns a tensor of shape (N, C, H, W) normalized to [-1, 1].
    """
    # Extract all jpg files and sort them alphabetically
    all_files = sorted([f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg'))])
    
    # Slice the list to get only the first X files
    if num_samples is not None:
        all_files = all_files[:num_samples]
        
    images = []
    
    # Setup transformations
    transform_list = []
    if image_size is not None:
        transform_list.append(transforms.Resize(image_size)) 
    transform_list.append(transforms.ToTensor()) # Converts to [0.0, 1.0] and shape (C, H, W)
    
    transform = transforms.Compose(transform_list) # Store the transformationto realize

    # Load, transform, and collect
    for f_name in all_files:
        file_path = os.path.join(root, f_name)
        try:
            img = Image.open(file_path).convert('RGB') # Make RGB
            img_tensor = transform(img) # Realize the list of transformations (resize maybe + to tensor)
            images.append(img_tensor)
        except Exception as e:
            print(f"Skipping {f_name} due to error: {e}")
            
    if not images:
        raise ValueError(f"No JPG images found in {root}")

    # Stack into a batch and scale from [0, 1] to [-1, 1]
    batch_tensor = torch.stack(images)
    batch_tensor = batch_tensor * 2 - 1 
    
    return batch_tensor.to('cpu')

import os
import torch
import torchvision.io as tv_io
import torchvision.transforms.functional as TF

def load_jpg_folder_torch(root='./data', num_samples=None, image_size=None, normalize=False,**kwargs):
    # Extract all jpg files and sort them alphabetically
    all_files = sorted([f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg'))])
    
    # Slice the list to get only the first X files
    if num_samples is not None:
        all_files = all_files[:num_samples]
        
    images = []
    
    # Load, transform, and collect
    for f_name in all_files:
        file_path = os.path.join(root, f_name)
        try:
            img_tensor = tv_io.decode_image(file_path, mode="RGB")

            if image_size is not None: # resize if specified
                img_tensor = TF.resize(img_tensor, image_size, antialias=True)
            
            images.append(img_tensor.float())
            
        except Exception as e:
            print(f"Skipping {f_name} due to error: {e}")
            
    if not images:
        raise ValueError(f"No JPG images found in {root}")

    batch_tensor = torch.stack(images)
    
    if normalize : batch_tensor = (batch_tensor / 255.0) * 2. - 1.
    
    return batch_tensor.to('cpu')

import io
import pandas as pd

@torch.no_grad()
def load_parquet(root="./data", num_samples=None, image_size=None, normalize=False,**kwargs):
    """
    - root : Can either be a filepath or a URL (based on doc but most of the time there're too much requests so it fails)
    """

    from tqdm import tqdm
    print('ahhhhhhhhhhh')
    df = pd.read_parquet(root)
    print("b")
    if num_samples is not None:
        df = df.head(num_samples)
    
    max_rows = num_samples if num_samples is not None else len(df.index)
    print(max_rows)
    if image_size is not None : 
        transform = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # TODO
        ])
    else : 
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if normalize : transform = transforms.Compose([*transform.transforms, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = []

    for _, row in tqdm(df.iterrows(), desc="Loading dataset", total=max_rows, miniters=1, mininterval=10):
        try:
            # Hugging Face stores images as a dictionary entry with a 'bytes' key
            img_data = row['image']
            img_bytes = img_data['bytes']
            
            # Convert bytes to a PIL image
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Skipping a row due to processing error: {e}")
            
    if not images:
        raise ValueError(f"No image in {root}")

    return torch.stack(images).to('cpu')

@torch.no_grad()
def load_parquet_attr(root="./data", num_samples=None, image_size=None, **kwargs):
    """
    - root : Can either be a filepath or a URL (based on doc but most of the time there're too much requests so it fails)
    """
    
    df = pd.read_parquet(root)
    
    if kwargs["target_labels"] != [] : 
        df = df [ df["label"]  == any(kwargs["target_labels"]) ]

    if num_samples is not None:
        df = df.head(num_samples)
    
    if image_size is not None : 
        transform = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # TODO
        ])
    else : 
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        
    
    images = []

    for _, row in df.iterrows():
        try:
            # Hugging Face stores images as a dictionary entry with a 'bytes' key
            img_data = row['image']
            img_bytes = img_data['bytes']
            
            # Convert bytes to a PIL image
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Skipping a row due to processing error: {e}")
            
    if not images:
        raise ValueError(f"No image in {root}")

    return torch.stack(images).to('cpu')
