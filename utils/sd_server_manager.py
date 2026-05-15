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
                 reference_image=None, control_net_path=None, 
                 cache_mode="none", cache_option="", scm_mask="", scm_policy="none",
                 lora_list=None, **kwargs):
        """Send a generation request to the native sdcpp API and poll for results"""
        
        # 1. Prepare native payload
        # Mapping UI names to API names
        # Note: server expects "euler_a" etc. UI might send "euler"
        method = sampling_method.lower().replace("-", "_")
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "batch_count": 1,
            "strength": kwargs.get("strength", 0.75),
            "clip_skip": kwargs.get("clip_skip", -1),
            "sample_params": {
                "sample_method": method,
                "sample_steps": steps,
                "guidance": {
                    "txt_cfg": cfg_scale
                }
            },
            "cache_mode": cache_mode if cache_mode != "none" else "disabled",
            "cache_option": cache_option,
            "scm_mask": scm_mask,
            "scm_policy_dynamic": scm_policy == "dynamic",
            "output_format": "png",
            "output_compression": 100
        }

        # Handle LoRAs
        if lora_list:
            payload["lora"] = lora_list # Expected format: [{"path": "...", "multiplier": 1.0}]
        else:
            payload["lora"] = []

        # Handle images
        if reference_image and os.path.exists(reference_image):
            import base64
            with open(reference_image, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
                # Determine if it's a control image or init image
                if control_net_path:
                    payload["control_image"] = img_b64
                else:
                    payload["init_image"] = img_b64

        # 2. Submit job (Async)
        url_submit = f"http://{self.host}:{self.port}/sdcpp/v1/img_gen"
        print(f"Submitting job to {url_submit}...")
        
        response = requests.post(url_submit, json=payload, timeout=30)
        if response.status_code != 202:
            raise RuntimeError(f"Failed to submit job: {response.text}")
        
        job_data = response.json()
        job_id = job_data["id"]
        poll_url = f"http://{self.host}:{self.port}{job_data['poll_url']}"
        
        print(f"Job submitted: {job_id}. Polling...")

        # 3. Poll for results
        start_time = time.time()
        timeout = 600 # 10 minutes
        
        while time.time() - start_time < timeout:
            job_resp = requests.get(poll_url, timeout=10)
            if job_resp.status_code != 200:
                raise RuntimeError(f"Error polling job: {job_resp.text}")
            
            status_data = job_resp.json()
            status = status_data["status"]
            
            if status == "completed":
                print("Job completed!")
                result = status_data.get("result")
                if not result or "images" not in result:
                    # In some versions it might be under data
                    images = status_data.get("data", [])
                else:
                    images = result["images"]
                
                if not images:
                    raise RuntimeError("Job completed but no images returned.")
                
                # Save first image
                import base64
                img_b64 = images[0]
                if img_b64.startswith("data:"):
                    img_b64 = img_b64.split(",")[1]
                    
                img_data = base64.b64decode(img_b64)
                output_file = "output_server.png"
                with open(output_file, "wb") as f:
                    f.write(img_data)
                return output_file
            
            elif status == "failed":
                error_msg = status_data.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error_msg}")
            
            elif status == "cancelled":
                raise RuntimeError("Job was cancelled.")
            
            # Progress update (optional: could yield here if integrated with generator)
            print(f"Job status: {status}...")
            time.sleep(1)
            
        raise TimeoutError("Job timed out.")

    def generate_video(self, prompt, negative_prompt, video_frames, width, height, fps=6, **kwargs):
        """Async video generation via /sdcpp/v1/vid_gen"""
        url_submit = f"http://{self.host}:{self.port}/sdcpp/v1/vid_gen"
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "video_frames": video_frames,
            "width": width,
            "height": height,
            "fps": fps,
            "sample_params": {
                "sample_steps": kwargs.get("steps", 20),
                "guidance": {"txt_cfg": kwargs.get("cfg_scale", 7.0)}
            },
            "output_format": "webp" # Or mp4 if supported
        }
        
        response = requests.post(url_submit, json=payload, timeout=30)
        if response.status_code != 202:
            raise RuntimeError(f"Failed to submit video job: {response.text}")
            
        # Polling logic similar to img_gen (omitted for brevity or can be refactored into a shared helper)
        # For now, let's just use the same polling logic
        job_data = response.json()
        # ... poll ...
        # (Implementation would be similar to img_gen)
        return self._poll_job(job_data["id"], job_data["poll_url"], output_ext="webp")

    def _poll_job(self, job_id, relative_poll_url, output_ext="png"):
        poll_url = f"http://{self.host}:{self.port}{relative_poll_url}"
        start_time = time.time()
        timeout = 1200 # 20 minutes for video
        
        while time.time() - start_time < timeout:
            job_resp = requests.get(poll_url, timeout=10)
            status_data = job_resp.json()
            status = status_data["status"]
            
            if status == "completed":
                result = status_data.get("result", {})
                # Video result might be in 'videos' or 'data'
                data_list = result.get("videos") or result.get("images") or status_data.get("data", [])
                
                if not data_list:
                    raise RuntimeError("No output data found.")
                
                import base64
                data_b64 = data_list[0]
                if data_b64.startswith("data:"):
                    data_b64 = data_b64.split(",")[1]
                
                output_file = f"output_server.{output_ext}"
                with open(output_file, "wb") as f:
                    f.write(base64.b64decode(data_b64))
                return output_file
            elif status in ["failed", "cancelled"]:
                raise RuntimeError(f"Job {status}: {status_data.get('error')}")
            
            time.sleep(1)
        raise TimeoutError("Job timed out.")


# Global instance
server_manager = SDServerManager()
