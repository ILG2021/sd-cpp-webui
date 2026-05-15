import subprocess
import os
import time
import requests
import json
import signal
from pathlib import Path


class SDServerManager:
    def __init__(self, cli_base_path=os.path.join("tools", "stable-diffusion-cpp"), host="127.0.0.1", port=8081):
        self.cli_base_path = cli_base_path
        self.host = host
        self.port = port
        self.process = None
        self.current_model_id = None

    def _get_server_exe(self):
        """Find the sd-server binary (cross-platform)"""
        ext = ".exe" if os.name == "nt" else ""
        exe_names = [f"sd-server{ext}", f"stable-diffusion-server{ext}"]

        for name in exe_names:
            paths = [
                os.path.join(self.cli_base_path, "bin", name),
                os.path.join(self.cli_base_path, "bin", "Release", name),
                os.path.join(self.cli_base_path, name)
            ]
            for p in paths:
                if os.path.exists(p):
                    return p
        return None

    def is_running(self):
        """Check if the server is alive and responding"""
        if self.process is None or self.process.poll() is not None:
            return False

        endpoints = ["/health", "/v1/models", "/"]
        for ep in endpoints:
            try:
                response = requests.get(f"http://{self.host}:{self.port}{ep}", timeout=1)
                if response.status_code == 200:
                    return True
            except:
                continue
        return False

    def stop(self):
        """Stop the server process"""
        if self.process:
            print(f"Stopping SD server (PID: {self.process.pid})...")
            try:
                # Use taskkill on Windows to ensure child processes are cleaned up
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], capture_output=True)
            except:
                self.process.terminate()
            self.process = None
            self.current_model_id = None

    def start(self, model_id, model_paths, params):
        """Start the server with specific model and parameters"""
        exe = self._get_server_exe()
        if not exe:
            raise FileNotFoundError("sd-server.exe not found. Please build the server target.")

        # If already running the same model, do nothing
        if self.is_running() and self.current_model_id == model_id:
            return True

        # Stop existing server if any
        self.stop()

        # Build command (similar to sd-cli but for server)
        cmd = [exe, "--listen-ip", self.host, "--listen-port", str(self.port)]

        # Add model paths (logic from cli_wrapper)
        for key, path in model_paths.items():
            if key == "diffusion-model":
                cmd.extend(["--diffusion-model", path])
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
            elif key == "taesd":
                cmd.extend(["--taesd", path])
            elif key == "tae":
                cmd.extend(["--tae", path])

        # Add performance/loading parameters (only those relevant to server startup)
        for key in ["diffusion-fa", "offload-to-cpu", "clip-on-cpu", "vae-on-cpu", "lora-model-dir"]:
            if key in params and params[key]:
                if isinstance(params[key], bool):
                    cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(params[key])])

        print(f"Starting SD server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            if self.is_running():
                self.current_model_id = model_id
                print("SD server is ready.")
                return True
            time.sleep(1)
            if self.process.poll() is not None:
                out, _ = self.process.communicate()
                raise RuntimeError(f"SD server failed to start:\n{out}")

        self.stop()
        raise TimeoutError("SD server timed out while starting.")

    def generate(self, prompt, negative_prompt, steps, cfg_scale, width, height, sampling_method, seed,
                 reference_image=None, control_net_path=None, **kwargs):
        """Send a generation request to the server"""
        url = f"http://{self.host}:{self.port}/v1/images/generations"

        # Construct payload with both standard OpenAI and sd-server specific fields
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "n": 1,
            "width": width,  # Some versions prefer explicit W/H
            "height": height,
            "size": f"{width}x{height}",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler": sampling_method,
            "sampling_method": sampling_method,
            "seed": seed,
            "response_format": "b64_json"
        }

        # Handle reference images (img2img/edit) via base64 if server supports it
        # Note: standard sd-server might need files via multipart or specific base64 fields
        if reference_image and os.path.exists(reference_image):
            import base64
            with open(reference_image, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
                payload["image"] = img_b64  # Common field for img2img in these servers
        print("url", url)
        print("paylaod",payload)
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            data = response.json()
            # Handle b64 data or save to file
            import base64
            img_data = base64.b64decode(data['data'][0]['b64_json'])
            output_file = "output.png"
            with open(output_file, "wb") as f:
                f.write(img_data)
            return output_file
        else:
            raise RuntimeError(f"Server error: {response.text}")


# Global instance
server_manager = SDServerManager()
