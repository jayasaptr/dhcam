import torch
import cv2
import time
from app.config import GPU_SETTINGS

def check_gpu_availability():
    """Check if GPU is available for PyTorch and OpenCV"""
    gpu_info = {
        'torch_cuda': False,
        'opencv_cuda': False,
        'device': 'cpu',
        'gpu_name': None,
        'gpu_memory': None
    }
    
    # Check PyTorch CUDA
    if torch.cuda.is_available():
        gpu_info['torch_cuda'] = True
        gpu_info['device'] = GPU_SETTINGS['device']
        gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
        gpu_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"✅ PyTorch CUDA available: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']:.1f} GB)")
    else:
        print("❌ PyTorch CUDA not available")
    
    # Check OpenCV CUDA
    try:
        cv2_build_info = cv2.getBuildInformation()
        if 'CUDA' in cv2_build_info and 'YES' in cv2_build_info:
            gpu_info['opencv_cuda'] = True
            print("✅ OpenCV with CUDA support detected")
        else:
            print("❌ OpenCV without CUDA support")
    except:
        print("❌ Could not check OpenCV CUDA support")
    
    return gpu_info

def setup_gpu_environment():
    """Setup GPU environment and memory management"""
    if not GPU_SETTINGS['enable_gpu']:
        print("🔧 GPU disabled in settings, using CPU")
        return 'cpu'
    
    gpu_info = check_gpu_availability()
    
    if gpu_info['torch_cuda']:
        device = gpu_info['device']
        
        # Set GPU memory fraction
        if GPU_SETTINGS['gpu_memory_fraction'] < 1.0:
            torch.cuda.set_per_process_memory_fraction(GPU_SETTINGS['gpu_memory_fraction'])
            print(f"🔧 GPU memory fraction set to {GPU_SETTINGS['gpu_memory_fraction']*100}%")
        
        # Enable mixed precision if supported
        if GPU_SETTINGS['mixed_precision']:
            print("🔧 Mixed precision enabled")
        
        print(f"🚀 Using GPU: {device}")
        return device
    else:
        print("⚠️ GPU not available, falling back to CPU")
        return 'cpu'

def optimize_model_for_gpu(model, device):
    """Optimize YOLO model for GPU inference"""
    if device != 'cpu':
        # Move model to GPU
        model.to(device)
        
        # Enable half precision if supported
        if GPU_SETTINGS['half_precision'] and device.startswith('cuda'):
            try:
                model.half()
                print("✅ Model converted to FP16 for faster inference")
            except Exception as e:
                print(f"⚠️ Could not convert to FP16: {e}")
        
        # Warm up the model
        print("🔄 Warming up GPU model...")
        import torch
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        if GPU_SETTINGS['half_precision']:
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            try:
                # This is a dummy inference to warm up the GPU
                _ = model.predict(dummy_input, verbose=False)
                print("✅ GPU model warmed up successfully")
            except:
                print("⚠️ GPU warmup failed, but model should still work")
    
    return model

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, cached
    return 0, 0

def print_gpu_stats():
    """Print GPU statistics"""
    if torch.cuda.is_available():
        allocated, cached = get_gpu_memory_info()
        print(f"📊 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    else:
        print("📊 GPU not available")

class GPUPerformanceMonitor:
    """Monitor GPU performance during inference"""
    def __init__(self):
        self.inference_times = []
        self.frame_count = 0
        self.start_time = None
        
    def start_inference(self):
        """Start timing inference"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        
    def end_inference(self):
        """End timing inference"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.start_time:
            inference_time = time.time() - self.start_time
            self.inference_times.append(inference_time)
            self.frame_count += 1
            
    def get_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return {}
            
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return {
            'avg_inference_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0,
            'total_frames': self.frame_count
        }
        
    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        if stats:
            print(f"🚀 GPU Performance - Avg: {stats['avg_inference_time']:.3f}s, FPS: {stats['fps']:.1f}")
