import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
        
        # Rename if necessary to match local_path
        if os.path.abspath(path) != os.path.abspath(local_path):
            os.rename(path, local_path)
            
        downloaded_paths[key] = local_path
        
    return downloaded_paths

def win_install_drivers():
    path1 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cublasLt64_12.dll'
    path2 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudnn_cnn64_9.dll'

    if not os.path.exists(path1) or not os.path.exists(path2):
        print("没有安装cudnn驱动，开始下载驱动")
        if not os.path.exists("cudnn.exe"):
            import gdown
            gdown.download("https://drive.google.com/file/d/1C0qJHToa72lLS2JUFO_89bYkKdW7ENB3/view?usp=drive_link",
                            "cudnn.exe",
                            fuzzy=True,
                            quiet=False)
        os.startfile("cudnn.exe")
    else:
        if os.path.exists("cudnn.exe"):
            os.remove("cudnn.exe")
