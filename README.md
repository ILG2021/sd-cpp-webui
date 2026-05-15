# Stable Diffusion C++ WebUI

这是一个基于 Gradio 构建的 `stable-diffusion.cpp` 图形化界面，支持多种前沿模型，并提供自动下载和服务器模式功能。

## 主要特性
- **模型支持**：Flux (1 & 2), Wan (视频), SD 3.5, Qwen, Chroma, Z-Image, Ovis, Anima 等。
- **服务器模式 (Server Mode)**：支持开启持久化服务器，将模型保持在内存/显存中，实现极速连续生成。
- **自动下载**：自动检测模型权重并从 HuggingFace 获取。
- **全面功能**：支持 LoRA, ControlNet (SD 1.5), PhotoMaker, ESRGAN 放大, TAESD 加速。
- **中文界面**：全中文汉化，更适合国内用户。
- **推理加速**：支持 Caching (ucache, dbcache, spectrum 等) 各种加速模式。

## 准备工作
1.  **Python 3.8+**
2.  **编译 stable-diffusion.cpp**：
    - 将编译好的项目放在 `tools/stable-diffusion.cpp/` 下。
    - 确保 `sd-cli.exe` 和 `sd-server.exe` 位于 `bin/` 或 `bin/Release/` 目录下。

## 安装与运行
```bash
# 安装依赖
pip install -r requirements.txt

## 运行应用
```powershell
# 1. 安装依赖 (详见下方 CUDA 说明)
pip install -r requirements.txt

# 2. 启动 WebUI
python app.py
```

## 显卡加速 (CUDA)
为了获得最佳性能，建议安装 CUDA 版本的 Python 绑定：
```powershell
$env:CMAKE_BUILD_PARALLEL_LEVEL = "24"  # 设置你的 CPU 核心数
$env:CMAKE_ARGS = "-DSD_CUDA=ON"
pip install stable-diffusion-cpp-python --force-reinstall --no-cache-dir -v
```

## 目录结构
- `app.py`: 主程序 (Gradio 界面)
- `utils/sd_python_manager.py`: 持久化模型管理器 (Python 绑定)
- `config/models.json`: 模型库配置文件
- `models/`: 模型权重存储目录 (自动下载)

## 进阶功能
- **持久化加载**：模型加载后会驻留在显存中，后续生成无需再次加载。切换模型时会自动更新。点击界面上的“卸载模型”可以手动释放显存。
- **LoRA**：支持在提示词中使用 `<lora:文件名:权重>` 语法。需在高级选项中正确配置 LoRA 目录。
