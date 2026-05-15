import subprocess
import os

def run_sd_cli(mode, params, model_paths, cli_base_path="tools/stable-diffusion.cpp"):
    """
    Constructs and runs the sd-cli command.
    """
    # Determine CLI path (checking both bin and bin/Release)
    cli_exe = os.path.join(cli_base_path, "bin", "sd-cli.exe")
    if not os.path.exists(cli_exe):
        cli_exe = os.path.join(cli_base_path, "bin", "Release", "sd-cli.exe")
    
    if not os.path.exists(cli_exe):
        return f"Error: sd-cli.exe not found at {cli_exe}. Please check the path."

    cmd = [cli_exe]
    
    # Add mode
    if mode == "vid_gen":
        cmd.extend(["-M", "vid_gen"])
    
    # Add model paths
    for key, path in model_paths.items():
        if key == "diffusion-model":
            cmd.extend(["--diffusion-model", path])
        elif key == "high-noise-diffusion-model":
            cmd.extend(["--high-noise-diffusion-model", path])
        elif key == "vae":
            cmd.extend(["--vae", path])
        elif key == "clip_l":
            cmd.extend(["--clip_l", path])
        elif key == "clip_g":
            cmd.extend(["--clip_g", path])
        elif key == "t5xxl":
            cmd.extend(["--t5xxl", path])
        elif key == "clip_vision":
            cmd.extend(["--clip_vision", path])
        elif key == "llm":
            cmd.extend(["--llm", path])
        elif key == "llm_vision":
            cmd.extend(["--llm_vision", path])
        elif key == "upscale-model":
            cmd.extend(["--upscale-model", path])
        elif key == "photo-maker":
            cmd.extend(["--photo-maker", path])
        elif key == "control-net":
            cmd.extend(["--control-net", path])
        elif key == "taesd":
            cmd.extend(["--taesd", path])
        elif key == "tae":
            cmd.extend(["--tae", path])

    # Add parameters
    for key, value in params.items():
        if value is not None and value != "":
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                # Handle single-dash parameters like -m, -p, -n, -i, -o, -r
                if len(key) == 1:
                    cmd.extend([f"-{key}", str(value)])
                else:
                    cmd.extend([f"--{key}", str(value)])

    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        # Run process and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        output_log = ""
        for line in process.stdout:
            print(line, end="")
            output_log += line
            yield output_log # Yielding for real-time progress in Gradio
            
        process.wait()
        
    except Exception as e:
        yield f"Error executing CLI: {str(e)}"
