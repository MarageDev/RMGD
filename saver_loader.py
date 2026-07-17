import os
import torch
import dataset_loader
# ls = load/save

def save_dict_hash(d:dict):
    """
    Return the dict (`d`) into a digest hash
    """
    
    from hashlib import sha256
    return sha256(str(d.items()).encode()).hexdigest()

def has_corresponding_reqs(tensor_dir_path:str, requirements:list=[]) -> tuple[bool, str]:
    """
    Searches for a file with specific requirements (hash, ...).
    Returns a tuplke:
    - If the file exists (bool)
    - The path to this file if found (else None)
    """
    for f in os.listdir(tensor_dir_path):
        c = f.split(".", 1)[0] # file name
        if all(req in str(c) for req in requirements):
            return (True, f)
    return (False, None)

@torch.no_grad()
def ls_with_cache_file_tensor(search_dir:str="./data/saved_tensors", preprocess_tensor_function:callable = None, tensor_to_save:torch.Tensor | dict = None, save_appendix:str=None,force_rewrite:bool = False, device="cpu", requirements:list = []) -> torch.Tensor:
    """
    Seach in `search_dir` for already existant file matching the requirements (`*requirements`). 
    - If a file is found, it is loaded.
    - If no file is found, it saves `tensor_to_save` and runs `preprocess_tensor_function` 
    (if not None, only ran if nothing is found, useful to avoid running heavy functions for nothing). The file name is composed of the search requirements (in order)
    """

    os.makedirs(search_dir, exist_ok=True)

    load_from_cache, existant_file_path = has_corresponding_reqs(search_dir, requirements)
    
    data_tensor = None
    
    can_load = not force_rewrite and load_from_cache
    
    if can_load:
        data_tensor = torch.load(f"{search_dir}/{existant_file_path}", map_location=device)
        print(f"Loaded tensor at {search_dir}/{existant_file_path}")
    else:
        data_tensor = tensor_to_save.to(device) if preprocess_tensor_function == None else preprocess_tensor_function(tensor_to_save.to(device))
        
        tensor_file_name = f"tensor_{(save_appendix + "_" )if all (save_appendix != i for i in [None, ""]) else ""}{"_".join(requirements)}.pt" 
        torch.save(data_tensor, f"{search_dir}/{tensor_file_name}")
        print(f"Saved tensor at {search_dir}/{tensor_file_name}")
        
    return data_tensor

@torch.no_grad()
def load_dataset(data_dir:str = "./data", dataset_loading_parameters:dict = None, device="cpu") -> torch.Tensor:
    
    """ 
    Load a tensor from the different datasets availible (listed below).
    
    Return [tensor, hash]
    
    Availible datasets :
    - celeba  (c)
    - cifar   (ci)
    - fashion (f)
    - art     (a)
    - cat     (ca)
    - dog     (d)
    - grumpy  (g)
    - obama   (o)
    - panda   (p)
    - flickr  (fl)
    - mnist   (m)
    - afhq    (af)
    - celebahq (chq)
    """
    
    data_set_name = dataset_loading_parameters["data_set_name"]

    root_appendix = None # mainly used for parquet loading, specifies where to load under .data/ if specified

    # Handle assigning the right dataset number to each datasets and its corresponding loading function to be used
    assigned_data_set = -1
    loading_func = None

    match data_set_name:
        case "celeba"   | "c"  : loading_func = dataset_loader.load_celeba          ;    root_appendix = None
        case "cifar"    | "ci" : loading_func = dataset_loader.load_cifar10         ;    root_appendix = None
        case "fashion"  | "f"  : loading_func = dataset_loader.load_mnist_fashion   ;    root_appendix = None
        case "art"      | "a"  : loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-art-painting"
        case "cat"      | "ca" : loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-cat"
        case "dog"      | "d"  : loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-dog"
        case "grumpy"   | "g"  : loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-grumpy-cat"
        case "obama"    | "o"  : loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-obama"
        case "panda"    | "p"  : loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-panda"
        case "flickr"   | "fl" : loading_func = dataset_loader.load_jpg_folder      ;    root_appendix = None
        case "mnist"    | "m"  : loading_func = dataset_loader.load_mnist           ;    root_appendix = None
        case "afhq"     | "af" : loading_func = dataset_loader.load_parquet_attr    ;    root_appendix = "afhq"
        case "celebahq" | "chq": loading_func = dataset_loader.load_parquet         ;    root_appendix = "celebahq"
        case "celebahq_jpg" | "chq_jpg": loading_func = dataset_loader.load_jpg_folder_torch         ;    root_appendix = "celebahq_jpg"
    
    dataset_loading_parameters["root"] = f"{data_dir}/{root_appendix}" if root_appendix else f"{data_dir}"

    data_tensor = loading_func(**dataset_loading_parameters).to(device)
    
    
    return data_tensor

@torch.no_grad()
def ls_with_cache_file_tensor_dataset(search_dir:str="./data/saved_tensors", data_dir:str = "./data",dataset_parameters:dict = None, preprocess_tensor_function:callable = None, force_rewrite:bool = False, save_appendix:str= "dataset" , requirements:list = [], device="cpu") -> torch.Tensor:
    """
    Seach in `search_dir` for already existant file matching the requirements and the hash from `dataset_parameters` (`requirements`). 
    - If a file is found, it is loaded.
    - If no file is found, it saves `tensor_to_save` and runs `preprocess_tensor_function` 
    (if not None, only ran if nothing is found, useful to avoid running heavy functions for nothing). The file name is composed of the search requirements (in order)
    
    Note : the file is saved in the searching directory.
    """

    os.makedirs(search_dir, exist_ok=True)

    current_hash = save_dict_hash(dataset_parameters)
    
    reqs = requirements + [current_hash]
    load_from_cache, existant_file_path = has_corresponding_reqs(search_dir, reqs)
    
    data_tensor = None
    
    can_load = not force_rewrite and load_from_cache
    
    if can_load:
        data_tensor = torch.load(f"{search_dir}/{existant_file_path}", map_location=device)
        print(f"Loaded dataset tensor at {search_dir}/{existant_file_path}")
    else:
        data_tensor = load_dataset(data_dir=data_dir, dataset_loading_parameters=dataset_parameters, device=device) if preprocess_tensor_function == None else preprocess_tensor_function(load_dataset(data_dir=data_dir, dataset_loading_parameters=dataset_parameters, device=device))
        
        tensor_file_name = f"tensor_{(save_appendix + "_" )if all (save_appendix != i for i in [None, ""]) else ""}{"_".join(reqs)}.pt" # Format : tensor_appendix_*requirements.pt
        torch.save(data_tensor, f"{search_dir}/{tensor_file_name}")
        print(f"Saved dataset tensor at {search_dir}/{tensor_file_name}")
        
    return data_tensor

