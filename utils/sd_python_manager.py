import os
import time
import gc
from PIL import Image

try:
    from stable_diffusion_cpp import StableDiffusion
except ImportError:
    StableDiffusion = None
    print("Warning: stable_diffusion_cpp not found. Python bindings mode will not work.")

class SDPythonManager:
    def __init__(self):
        self.instance = None
        self.current_model_id = None
        self.current_params = {}

    def is_available(self):
        return StableDiffusion is not None

    def load_model(self, model_id, model_config, lora_model_dir=None):
        if not self.is_available():
            raise ImportError("stable-diffusion-cpp-python package is not installed.")

        # Check if we already have this model loaded with the same critical parameters
        if self.current_model_id == model_id and self.instance is not None:
            # We assume if the model_id is the same, the underlying files haven't changed meaningfully
            # for a simple session.
            print(f"Model {model_id} already loaded in memory.")
            return True

        print(f"Loading model: {model_id} into memory...")
        files = model_config.get("files", {})
        def_params = model_config.get("default_params", {})
        
        kwargs = {}
        
        # Mapping files to StableDiffusion constructor arguments
        # Main model
        if "diffusion-model" in files:
            path = files["diffusion-model"]["local_path"]
            # Detect if it's a "standard" SD model or a FLUX/Wan model that needs diffusion_model_path
            # Heuristic: if it has t5xxl or clip_l, it's likely FLUX/Wan/SD3
            if any(k in files for k in ["t5xxl", "clip_l", "llm"]):
                kwargs["diffusion_model_path"] = path
            else:
                kwargs["model_path"] = path

        if "vae" in files:
            kwargs["vae_path"] = files["vae"]["local_path"]
        if "clip_l" in files:
            kwargs["clip_l_path"] = files["clip_l"]["local_path"]
        if "clip_g" in files:
            kwargs["clip_g_path"] = files["clip_g"]["local_path"]
        if "t5xxl" in files:
            kwargs["t5xxl_path"] = files["t5xxl"]["local_path"]
        if "llm" in files:
            kwargs["llm_path"] = files["llm"]["local_path"]
        if "clip_vision" in files:
            kwargs["clip_vision_path"] = files["clip_vision"]["local_path"]
        if "control-net" in files:
            kwargs["control_net_path"] = files["control-net"]["local_path"]
            
        if lora_model_dir:
            kwargs["lora_model_dir"] = lora_model_dir

        # Hardware/Performance options
        if def_params.get("diffusion-fa"):
            kwargs["diffusion_flash_attn"] = True
        if def_params.get("offload-to-cpu"):
            kwargs["offload_params_to_cpu"] = True
        if def_params.get("clip-on-cpu"):
            kwargs["keep_clip_on_cpu"] = True
        if def_params.get("vae-on-cpu"):
            kwargs["keep_vae_on_cpu"] = True

        # Unload previous model if exists
        if self.instance:
            print("Unloading previous model...")
            self.instance = None
            gc.collect()
            # If using CUDA, we might want to clear torch cache if it was used, 
            # but stable-diffusion.cpp uses its own backend.
            
        try:
            self.instance = StableDiffusion(**kwargs)
            self.current_model_id = model_id
            self.current_params = kwargs
            print(f"Model {model_id} loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load model {model_id}: {str(e)}")
            self.current_model_id = None
            self.instance = None
            raise e

    def generate(self, prompt, negative_prompt, steps, cfg_scale, width, height, sample_method, seed, 
                 init_image=None, mask_image=None, strength=0.75, **kwargs):
        if not self.instance:
            yield "错误：模型未加载。请先选择模型并加载。"
            return

        yield "开始生成 (Python 绑定模式)..."
        
        # Clean up sample method name
        method = sample_method.lower().replace("-", "_")
        
        try:
            # Prepare arguments
            gen_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "sample_steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "sample_method": method,
                "seed": seed,
            }

            if init_image:
                gen_args["init_image"] = init_image
                gen_args["strength"] = strength
            
            if mask_image:
                gen_args["mask_image"] = mask_image

            # Check for other special kwargs
            if "clip_skip" in kwargs:
                gen_args["clip_skip"] = kwargs["clip_skip"]

            # Generate
            start_time = time.time()
            output = self.instance.generate_image(**gen_args)
            end_time = time.time()
            
            if output and len(output) > 0:
                # Save image
                os.makedirs("outputs", exist_ok=True)
                timestamp = int(time.time())
                filename = os.path.join("outputs", f"out_{timestamp}.png")
                output[0].save(filename)
                yield f"生成完成，用时 {end_time - start_time:.2f}s"
                yield filename
            else:
                yield "生成失败：返回结果为空。"
        except Exception as e:
            yield f"生成过程中发生异常: {str(e)}"

    def generate_video(self, prompt, negative_prompt, video_frames, width, height, steps=20, cfg_scale=6.0, **kwargs):
        if not self.instance:
            yield "错误：模型未加载。"
            return

        yield "开始视频生成 (Python 绑定模式)..."
        
        try:
            # Based on library docs, video generation might be a separate method or args
            # The README says "Wan image/video generation"
            # Assuming it's generate_video if available, or generate_image with video_frames
            if hasattr(self.instance, "generate_video"):
                output = self.instance.generate_video(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    video_frames=video_frames,
                    sample_steps=steps,
                    cfg_scale=cfg_scale,
                    **kwargs
                )
            else:
                # Fallback to generate_image if it handles video_frames
                output = self.instance.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    video_frames=video_frames,
                    sample_steps=steps,
                    cfg_scale=cfg_scale,
                    **kwargs
                )
            
            if output and len(output) > 0:
                os.makedirs("outputs", exist_ok=True)
                timestamp = int(time.time())
                # If output is a list of images (frames), we might need to encode it
                # Or if the library returns a video file path/bytes
                if isinstance(output, list) and isinstance(output[0], Image.Image):
                    # It's frames
                    filename = os.path.join("outputs", f"video_{timestamp}.webp")
                    output[0].save(filename, save_all=True, append_images=output[1:], duration=100, loop=0)
                    yield filename
                elif isinstance(output, str):
                    # It might be a path to a generated mp4
                    yield output
                else:
                    yield "视频生成成功，但返回格式无法识别。"
            else:
                yield "视频生成失败。"
        except Exception as e:
            yield f"视频生成出错: {str(e)}"

# Global singleton
python_manager = SDPythonManager()
