import gradio as gr
import json
import os
import time
from utils.downloader import download_model
from utils.cli_wrapper import run_sd_cli
from utils.sd_server_manager import server_manager

# 加载模型配置
with open("config/models.json", "r") as f:
    MODELS_CONFIG = json.load(f)["models"]

IMAGE_MODELS = {k: v for k, v in MODELS_CONFIG.items() if v["type"] != "vid_gen"}
VIDEO_MODELS = {k: v for k, v in MODELS_CONFIG.items() if v["type"] == "vid_gen"}

def generate_call(model_name, prompt, negative_prompt, steps, cfg_scale, width, height, sampling_method, seed, 
                  diffusion_fa, offload_to_cpu, clip_on_cpu, vae_on_cpu, lora_model_dir, 
                  taesd_path, tae_path, cache_mode, cache_option, scm_mask, scm_policy,
                  video_frames, flow_shift, reference_image, control_net_path, upscale_model, 
                  pm_images_dir, pm_style_strength, use_server_mode):
    
    if model_name not in MODELS_CONFIG:
        return None, "错误：未找到所选模型的配置。"

    config = MODELS_CONFIG[model_name]
    defaults = config.get("default_params", {})
    
    # 1. 自动权重下载
    yield None, f"正在检查并下载 {model_name} 的权重文件..."
    try:
        model_paths = download_model(config)
    except Exception as e:
        yield None, f"下载权重时出错: {str(e)}"
        return

    # 2. 准备基础参数
    params = {
        "p": prompt,
        "n": negative_prompt,
        "steps": steps,
        "cfg-scale": cfg_scale,
        "W": width,
        "H": height,
        "sampling-method": sampling_method,
        "seed": seed if seed != -1 else int(time.time()),
        "diffusion-fa": diffusion_fa,
        "offload-to-cpu": offload_to_cpu,
        "clip-on-cpu": clip_on_cpu,
        "vae-on-cpu": vae_on_cpu,
        "lora-model-dir": lora_model_dir,
        "cache-mode": cache_mode if cache_mode != "none" else None,
        "cache-option": cache_option,
        "scm-mask": scm_mask,
        "scm-policy": scm_policy if scm_policy != "none" else None,
        "high-noise-cfg-scale": defaults.get("high-noise-cfg-scale"),
        "high-noise-steps": defaults.get("high-noise-steps"),
        "high-noise-sampling-method": defaults.get("high-noise-sampling-method"),
        "chroma-disable-dit-mask": defaults.get("chroma-disable-dit-mask"),
    }
    
    if taesd_path: model_paths["taesd"] = taesd_path
    if tae_path: model_paths["tae"] = tae_path

    # --- 服务器模式处理 ---
    if use_server_mode and config["type"] != "vid_gen":
        yield None, f"正在启动服务器模式并加载模型 {model_name} (这可能需要一些时间)..."
        try:
            server_manager.start(model_name, model_paths, params)
            yield None, "服务器已就绪，正在发送生成请求..."
            output_file = server_manager.generate(
                prompt, negative_prompt, steps, cfg_scale, width, height, sampling_method, params["seed"],
                reference_image=reference_image, control_net_path=control_net_path
            )
            yield output_file, "生成成功 (服务器模式)。"
            return
        except Exception as e:
            yield None, f"服务器模式失败，回退到 CLI 模式: {str(e)}"
            # Fallback to CLI
    
    # --- 标准 CLI 模式 ---
    if config["type"] == "vid_gen":
        params["video-frames"] = video_frames
        params["flow-shift"] = flow_shift
        output_file = "output.webm"
    else:
        output_file = "output.png"
    
    if reference_image is not None:
        params["r"] = reference_image
        if "Stable-Diffusion" in model_name:
            if control_net_path:
                params["control-image"] = reference_image
                model_paths["control-net"] = control_net_path
            else:
                params["i"] = reference_image
                params["strength"] = 0.7

    if upscale_model:
        params["upscale-model"] = upscale_model

    if pm_images_dir:
        params["pm-id-images-dir"] = pm_images_dir
        params["pm-style-strength"] = pm_style_strength

    params["o"] = output_file

    yield None, "正在通过 CLI 开始生成..."
    log_output = ""
    for log in run_sd_cli(config["type"], params, model_paths):
        log_output = log
        yield None, log_output

    if os.path.exists(output_file):
        yield output_file, log_output
    else:
        yield None, log_output + "\n错误：未生成输出文件。"

def create_gen_tab(models_dict, is_video=False):
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                model_name = gr.Dropdown(choices=list(models_dict.keys()), value=list(models_dict.keys())[0], label="选择模型", scale=3)
                use_server_mode = gr.Checkbox(label="开启服务器模式 (保持模型在内存中)", value=False, scale=1, visible=not is_video)
            
            prompt = gr.Textbox(label="提示词 (Prompt)", placeholder="输入你想要生成的画面描述...", lines=3)
            negative_prompt = gr.Textbox(label="反向提示词 (Negative Prompt)", placeholder="输入你不想要出现的元素...", lines=2)
            
            with gr.Tabs():
                with gr.Tab("基础设置"):
                    with gr.Row():
                        steps = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="步数 (Steps)")
                        cfg_scale = gr.Slider(minimum=0.1, maximum=20.0, step=0.1, value=1.0, label="提示词相关性 (CFG Scale)")
                    with gr.Row():
                        width = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="宽度")
                        height = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="高度")
                    sampling_method = gr.Dropdown(choices=["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2m_v2", "lcm"], value="euler", label="采样方法")
                    seed = gr.Number(value=-1, label="随机种子 (Seed, -1为随机)")

                with gr.Tab("参考图/视频"):
                    reference_image = gr.Image(label="参考图 (用于编辑/图生图/图生视频/ControlNet)", type="filepath")
                    control_net_path = gr.Textbox(label="ControlNet 模型路径", placeholder="models/control_v11p_sd15_canny.safetensors")
                    if is_video:
                        video_frames = gr.Slider(minimum=1, maximum=121, step=1, value=33, label="视频总帧数")
                        flow_shift = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="流偏移 (Flow Shift)")
                    else:
                        video_frames = gr.State(33)
                        flow_shift = gr.State(3.0)

                with gr.Tab("推理加速 (Caching)"):
                    cache_mode = gr.Dropdown(choices=["none", "ucache", "easycache", "dbcache", "taylorseer", "cache-dit", "spectrum"], value="none", label="缓存模式")
                    cache_option = gr.Textbox(label="缓存选项", placeholder="例如: threshold=1.0,warmup=4")
                    scm_mask = gr.Textbox(label="SCM 掩码 (SCM Mask)", placeholder="例如: 1,1,0,1...")
                    scm_policy = gr.Dropdown(choices=["none", "dynamic", "static"], value="none", label="SCM 策略")

                with gr.Tab("性能/高级选项"):
                    with gr.Row():
                        diffusion_fa = gr.Checkbox(label="闪电注意力 (Flash Attention)", value=True)
                        offload_to_cpu = gr.Checkbox(label="权重卸载至 CPU", value=False)
                    with gr.Row():
                        clip_on_cpu = gr.Checkbox(label="CLIP 在 CPU 运行", value=False)
                        vae_on_cpu = gr.Checkbox(label="VAE 在 CPU 运行", value=False)
                    
                    lora_model_dir = gr.Textbox(label="LoRA 模型目录", placeholder="例如: models/loras")
                    upscale_model = gr.Textbox(label="放大模型路径 (ESRGAN)", placeholder="例如: models/realesrgan.pth")
                    taesd_path = gr.Textbox(label="TAESD 路径", placeholder="models/taesd.safetensors")
                    tae_path = gr.Textbox(label="TAE 路径 (视频专用)", placeholder="models/taew2_1.safetensors")
                
                if not is_video:
                    with gr.Tab("人物写真 (PhotoMaker)"):
                        pm_images_dir = gr.Textbox(label="人物图片目录", placeholder="包含同一个人的多张照片的文件夹路径")
                        pm_style_strength = gr.Slider(minimum=0, maximum=100, step=1, value=20, label="风格强度 (%)")
                else:
                    pm_images_dir = gr.State("")
                    pm_style_strength = gr.State(20)

            with gr.Row():
                generate_btn = gr.Button("🔥 立即生成", variant="primary", scale=3)
                stop_server_btn = gr.Button("⏹️ 停止服务器", variant="secondary", scale=1, visible=not is_video)

        with gr.Column(scale=1):
            output_display = gr.File(label="生成结果")
            log_display = gr.Textbox(label="运行日志", lines=20, interactive=False)

    def update_model_defaults(m_name):
        config = models_dict[m_name]
        defaults = config.get("default_params", {})
        return {
            steps: gr.update(value=defaults.get("steps", 20)),
            cfg_scale: gr.update(value=defaults.get("cfg-scale", 1.0)),
            width: gr.update(value=defaults.get("W", 1024)),
            height: gr.update(value=defaults.get("H", 1024)),
            diffusion_fa: gr.update(value=defaults.get("diffusion-fa", True)),
            offload_to_cpu: gr.update(value=defaults.get("offload-to-cpu", False)),
            clip_on_cpu: gr.update(value=defaults.get("clip-on-cpu", False)),
        }

    model_name.change(update_model_defaults, inputs=[model_name], outputs=[steps, cfg_scale, width, height, diffusion_fa, offload_to_cpu, clip_on_cpu])

    def stop_server_action():
        server_manager.stop()
        return "服务器已停止，显存已释放。"

    if not is_video:
        stop_server_btn.click(stop_server_action, outputs=[log_display])

    generate_btn.click(
        generate_call,
        inputs=[
            model_name, prompt, negative_prompt, steps, cfg_scale, width, height, sampling_method, seed, 
            diffusion_fa, offload_to_cpu, clip_on_cpu, vae_on_cpu, lora_model_dir, 
            taesd_path, tae_path, cache_mode, cache_option, scm_mask, scm_policy,
            video_frames, flow_shift, reference_image, control_net_path, upscale_model, pm_images_dir, pm_style_strength,
            use_server_mode
        ],
        outputs=[output_display, log_display]
    )

# Gradio UI 主架构
with gr.Blocks(title="SD.cpp 桌面版 WebUI") as demo:
    gr.Markdown("# 🚀 Stable Diffusion C++ 全功能 WebUI")
    
    with gr.Tabs():
        with gr.Tab("🖼️ 图片生成"):
            create_gen_tab(IMAGE_MODELS, is_video=False)
            
        with gr.Tab("🎬 视频生成"):
            create_gen_tab(VIDEO_MODELS, is_video=True)

if __name__ == "__main__":
    demo.queue().launch()
