import os
import torch

import dataset_loader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##### MAIN PARAMETERS

def load_data_to_tensor(override_path:str, data_dir:str = "./data", dataset_loading_parameters:dict = None, force_reload_tensor=True) -> torch.Tensor:
    """ Availible datasets :
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
    """
    data_set_name = dataset_loading_parameters["data_set_name"]
    
    if override_path is not None and override_path != "" : 
        print(f"Loading tensor from : {override_path}")
        return torch.load(override_path, map_location=device)
    
    root_appendix = None # mainly used for parquet loading, specifies where to load under .data/ if specified

    # Handle assigning the right dataset number to each datasets and its corresponding loading function to be used
    assigned_data_set = -1
    loading_func = None

    match data_set_name:
        case "celeba"   | "c"  : assigned_data_set = 0     ;   loading_func = dataset_loader.load_celeba          ;    root_appendix = None
        case "cifar"    | "ci" : assigned_data_set = 1     ;   loading_func = dataset_loader.load_cifar10         ;    root_appendix = None
        case "fashion"  | "f"  : assigned_data_set = 2     ;   loading_func = dataset_loader.load_mnist_fashion   ;    root_appendix = None
        case "art"      | "a"  : assigned_data_set = 3     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-art-painting"
        case "cat"      | "ca" : assigned_data_set = 4     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-cat"
        case "dog"      | "d"  : assigned_data_set = 5     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-dog"
        case "grumpy"   | "g"  : assigned_data_set = 6     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-grumpy-cat"
        case "obama"    | "o"  : assigned_data_set = 7     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-obama"
        case "panda"    | "p"  : assigned_data_set = 8     ;   loading_func = dataset_loader.load_parquet         ;    root_appendix = "few-shot-panda"
        case "flickr"   | "fl" : assigned_data_set = 9     ;   loading_func = dataset_loader.load_jpg_folder      ;    root_appendix = None
        case "mnist"    | "m"  : assigned_data_set = 10    ;   loading_func = dataset_loader.load_mnist           ;    root_appendix = None
        case "afhq"     | "af" : assigned_data_set = 11    ;   loading_func = dataset_loader.load_parquet_attr    ;    root_appendix = "afhq"

    dataset_loading_parameters["root"] = f"{data_dir}/{root_appendix}" if root_appendix else f"{data_dir}"
    
    # loading/saving of cahce
    cache_dir = f"{data_dir}/saved_tensors"
    os.makedirs(cache_dir, exist_ok=True)
    
    load_from_cache = False
    search_result = ()
    current_param_hash = save_dict_hash(dataset_loading_parameters)
    if not force_reload_tensor:
        search_result = has_corresponding_tensor_hash(f"{data_dir}/saved_tensors",current_param_hash)
        load_from_cache = search_result[0]
        
    if load_from_cache: # Load tensor from an already created cached tensor file
        print(f"Loading tensor from : {search_result[1]}")
        data_tensor = torch.load(f"{cache_dir}/{search_result[1]}", map_location=device)
    else: # Create a new cache tensor file
        print(f"Processing dataset {data_set_name} ({assigned_data_set}) and creating tensor file (tensor_cache_{assigned_data_set}_{str(current_param_hash)}.pt)")
        
        data_tensor = loading_func(**dataset_loading_parameters)
        # Save the tensor and update the file
        
        tensor_file_name = f"tensor_cache_{assigned_data_set}_{str(current_param_hash)}.pt"
        
        print(f"{cache_dir}/{tensor_file_name}")
        torch.save(data_tensor, f"{cache_dir}/{tensor_file_name}")
    associated_hash = current_param_hash
    return data_tensor, associated_hash


def has_corresponding_tensor_hash(tensor_dir_path, target_hash, additional_req=None) -> tuple:
    # Try to retrieve a saved tensor path with the corresponding parameter hash
    import os
    if additional_req == None :
        for f in os.listdir(tensor_dir_path):
            c = [ i.split(".") for i in f.split("_")][-1][0] #retrieve hash, last list, first item
            if str(c) == str(target_hash):
                return (True, f)
    else : 
        for f in os.listdir(tensor_dir_path):
            c = [ i.split(".") for i in f.split("_")][-1][0] #retrieve hash, last list, first item
            if str(c) == str(target_hash) and additional_req in f:
                return (True, f)
    return (False, None)

def save_or_load_if_hash(tensor,data_dir="./data",hash=None, device="cuda"):
    cache_dir = f"{data_dir}/saved_tensors"
    os.makedirs(cache_dir, exist_ok=True)
    
    load_from_cache = False
    search_result = ()
    current_param_hash = hash
    if load_from_cache: # Load tensor from an already created cached tensor file
        data_tensor = torch.load(f"{cache_dir}/{search_result[1]}", map_location=device)
        return data_tensor
    else: # Create a new cache tensor file
        tensor_file_name = f"tensor_cache_encoded_{str(current_param_hash)}.pt"
        torch.save(data_tensor, f"{cache_dir}/{tensor_file_name}")
        return data_tensor
    
def save_dict_hash(d):
    from hashlib import sha256
    return sha256(str(d.items()).encode()).hexdigest()