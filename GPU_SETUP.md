# GPU Acceleration Setup

This guide will help you set up GPU acceleration for the Hard Hat Detection System to improve performance and reduce camera delay.

## 🚀 Performance Benefits

With GPU acceleration, you can expect:

- **3-5x faster inference** compared to CPU
- **Reduced camera delay** from buffering
- **Higher FPS processing** capability
- **Lower CPU usage**

## 📋 Prerequisites

1. **NVIDIA GPU** with CUDA support (GTX 1060 or better recommended)
2. **NVIDIA Drivers** (latest version)
3. **CUDA Toolkit** (11.8 or 12.x)

## 🔧 Installation Steps

### Step 1: Check GPU Compatibility

```bash
# Check if NVIDIA GPU is detected
nvidia-smi
```

### Step 2: Automatic Setup

```bash
# Run the automated setup script
python setup_gpu.py
```

### Step 3: Manual Setup (if needed)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### Step 4: Test GPU Setup

```bash
# Test GPU functionality
python test_gpu.py
```

## ⚙️ Configuration

GPU settings can be modified in `app/config.py`:

```python
GPU_SETTINGS = {
    'enable_gpu': True,              # Enable/disable GPU acceleration
    'device': 'cuda:0',              # GPU device (cuda:0 for first GPU)
    'use_cuda_opencv': True,         # Use CUDA for OpenCV operations
    'gpu_memory_fraction': 0.8,      # GPU memory usage limit (80%)
    'mixed_precision': True,         # Enable mixed precision training
    'batch_size': 1,                 # Inference batch size
    'half_precision': True,          # Use FP16 for faster inference
}
```

## 🔍 Performance Monitoring

The system includes built-in GPU monitoring:

- **Memory usage tracking**
- **Inference time measurement**
- **FPS calculation**
- **Performance statistics**

Monitor output will show:

```
✅ PyTorch CUDA available: NVIDIA GeForce RTX 3080 (10.0 GB)
✅ Model converted to FP16 for faster inference
🚀 GPU Performance - Avg: 0.045s, FPS: 22.2
📊 GPU Memory - Allocated: 2.34GB, Cached: 2.50GB
```

## 🛠️ Troubleshooting

### Common Issues:

**1. CUDA Out of Memory**

```python
# Reduce GPU memory fraction
GPU_SETTINGS['gpu_memory_fraction'] = 0.6
```

**2. Model Not Using GPU**

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**3. OpenCV CUDA Issues**

```bash
# Install OpenCV contrib
pip install opencv-contrib-python
```

**4. Slow Performance**

```python
# Enable optimizations
GPU_SETTINGS['half_precision'] = True
GPU_SETTINGS['mixed_precision'] = True
```

### Performance Tuning:

**For High-End GPUs (RTX 3080+):**

```python
GPU_SETTINGS['gpu_memory_fraction'] = 0.9
GPU_SETTINGS['half_precision'] = True
```

**For Mid-Range GPUs (GTX 1060-1080):**

```python
GPU_SETTINGS['gpu_memory_fraction'] = 0.7
GPU_SETTINGS['half_precision'] = False
```

**For Low VRAM GPUs (<6GB):**

```python
GPU_SETTINGS['gpu_memory_fraction'] = 0.5
GPU_SETTINGS['half_precision'] = False
```

## 📈 Expected Performance

| GPU Model | Inference Time | FPS | Memory Usage |
| --------- | -------------- | --- | ------------ |
| RTX 4090  | ~20ms          | 50+ | 3-4GB        |
| RTX 3080  | ~35ms          | 28+ | 2-3GB        |
| RTX 2070  | ~50ms          | 20+ | 2-3GB        |
| GTX 1660  | ~80ms          | 12+ | 2GB          |

## 🔄 Fallback to CPU

If GPU acceleration fails, the system automatically falls back to CPU processing:

```
⚠️ GPU not available, falling back to CPU
ℹ️ Running on CPU - GPU not available
```

## 💡 Tips for Best Performance

1. **Close unnecessary applications** to free GPU memory
2. **Use FP16 precision** for 2x speed improvement on newer GPUs
3. **Monitor GPU temperature** to prevent throttling
4. **Update NVIDIA drivers** regularly
5. **Use dedicated GPU** (not integrated graphics)

## 🆘 Support

If you encounter issues:

1. Run `python test_gpu.py` for diagnostics
2. Check GPU temperature and usage with `nvidia-smi`
3. Verify CUDA installation with `nvcc --version`
4. Check PyTorch CUDA with `python -c "import torch; print(torch.cuda.is_available())"`
