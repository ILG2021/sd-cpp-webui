import os
from huggingface_hub import hf_hub_download

def download_model(model_config):
    """
    Downloads model weights based on the provided configuration.
    """
    downloaded_paths = {}
    
    for key, file_info in model_config['files'].items():
        repo_id = file_info['repo']
        filename = file_info['filename']
        subfolder = file_info.get('subfolder', None)
        local_path = file_info['local_path']
        
        # Check if file already exists
        if os.path.exists(local_path):
            print(f"File {local_path} already exists. Skipping download.")
            downloaded_paths[key] = local_path
            continue
            
        print(f"Downloading {filename} from {repo_id}...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download from HF
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False
        )
        
        # Rename if necessary to match local_path
        if os.path.abspath(path) != os.path.abspath(local_path):
            os.rename(path, local_path)
            
        downloaded_paths[key] = local_path
        
    return downloaded_paths
