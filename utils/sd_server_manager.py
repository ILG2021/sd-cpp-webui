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
        """Send a generation request to the native sdcpp API and poll for results (Generator)"""
        
        # 1. Prepare native payload
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

        if lora_list:
            payload["lora"] = lora_list
        else:
            payload["lora"] = []

        if reference_image and os.path.exists(reference_image):
            import base64
            with open(reference_image, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
                if control_net_path:
                    payload["control_image"] = img_b64
                else:
                    payload["init_image"] = img_b64

        # 2. Submit job (Async)
        url_submit = f"http://{self.host}:{self.port}/sdcpp/v1/img_gen"
        yield f"正在提交任务到 {url_submit}..."
        
        try:
            response = requests.post(url_submit, json=payload, timeout=30)
            if response.status_code != 202:
                yield f"服务器报错: {response.text}"
                raise RuntimeError(f"Failed to submit job: {response.text}")
        except Exception as e:
            yield f"请求异常: {str(e)}"
            raise e
        
        job_data = response.json()
        job_id = job_data["id"]
        poll_url = f"http://{self.host}:{self.port}{job_data['poll_url']}"
        
        yield f"任务已提交 (ID: {job_id})，正在等待生成..."

        # 3. Poll for results
        for update in self._poll_job(job_id, job_data["poll_url"], output_ext="png"):
            yield update

    def generate_video(self, prompt, negative_prompt, video_frames, width, height, fps=6, **kwargs):
        """Async video generation via /sdcpp/v1/vid_gen (Generator)"""
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
            "output_format": "webp"
        }
        
        yield f"正在提交视频任务到 {url_submit}..."
        response = requests.post(url_submit, json=payload, timeout=30)
        if response.status_code != 202:
            yield f"服务器报错: {response.text}"
            raise RuntimeError(f"Failed to submit video job: {response.text}")
            
        job_data = response.json()
        yield f"视频任务已提交 (ID: {job_data['id']})，正在等待生成..."
        
        for update in self._poll_job(job_data["id"], job_data["poll_url"], output_ext="webp"):
            yield update

    def _poll_job(self, job_id, relative_poll_url, output_ext="png"):
        poll_url = f"http://{self.host}:{self.port}{relative_poll_url}"
        start_time = time.time()
        timeout = 1200
        
        while time.time() - start_time < timeout:
            try:
                job_resp = requests.get(poll_url, timeout=10)
                status_data = job_resp.json()
                status = status_data["status"]
                
                if status == "completed":
                    yield "任务已完成，正在接收数据..."
                    result = status_data.get("result", {})
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
                    yield output_file
                    return
                
                elif status in ["failed", "cancelled"]:
                    err = status_data.get("error", "未知错误")
                    yield f"任务 {status}: {err}"
                    raise RuntimeError(f"Job {status}: {err}")
                
                yield f"当前状态: {status}..."
            except Exception as e:
                yield f"轮询异常: {str(e)}"
            
            time.sleep(1)
        raise TimeoutError("Job timed out.")


# Global instance
server_manager = SDServerManager()
