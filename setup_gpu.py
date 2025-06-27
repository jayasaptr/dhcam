#!/usr/bin/env python3
"""
GPU Setup Script for Hard Hat Detection System
This script helps set up GPU acceleration for the system
"""

import subprocess
import sys
import platform

def check_cuda_installation():
    """Check if CUDA is installed"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print(result.stdout.split('\n')[2])  # GPU info line
            return True
        else:
            print("❌ nvidia-smi not found")
            return False
    except FileNotFoundError:
        print("❌ NVIDIA drivers not installed")
        return False

def install_torch_gpu():
    """Install PyTorch with CUDA support"""
    print("🔄 Installing PyTorch with CUDA support...")
    
    # For CUDA 11.8
    cuda_version = "cu118"
    torch_url = f"https://download.pytorch.org/whl/{cuda_version}"
    
    commands = [
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", 
         "--index-url", torch_url],
        [sys.executable, "-m", "pip", "install", "ultralytics"],
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Failed to install {cmd[3]}")
            return False
    
    print("✅ PyTorch with CUDA installed")
    return True

def install_opencv_cuda():
    """Install OpenCV with CUDA support"""
    print("🔄 Installing OpenCV with CUDA support...")
    
    # Note: This typically requires building from source
    # For now, we'll install the contrib version which may have some CUDA support
    cmd = [sys.executable, "-m", "pip", "install", "opencv-contrib-python"]
    
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ OpenCV contrib installed")
        print("ℹ️ For full CUDA support, you may need to build OpenCV from source")
        return True
    else:
        print("❌ Failed to install OpenCV contrib")
        return False

def main():
    print("🚀 GPU Setup for Hard Hat Detection")
    print("=" * 50)
    
    # Check system
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Check CUDA
    cuda_available = check_cuda_installation()
    
    if not cuda_available:
        print("\n⚠️ CUDA not detected. Please install:")
        print("1. NVIDIA GPU drivers")
        print("2. CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
        return
    
    print("\n🔄 Setting up GPU packages...")
    
    # Install PyTorch GPU
    torch_success = install_torch_gpu()
    
    # Install OpenCV
    opencv_success = install_opencv_cuda()
    
    print("\n📋 Setup Summary:")
    print("-" * 30)
    print(f"CUDA Available: {'✅' if cuda_available else '❌'}")
    print(f"PyTorch GPU: {'✅' if torch_success else '❌'}")
    print(f"OpenCV: {'✅' if opencv_success else '❌'}")
    
    if torch_success:
        print("\n🎉 GPU setup completed!")
        print("\n💡 Next steps:")
        print("1. Run: python test_gpu.py")
        print("2. If tests pass, run: python main.py")
    else:
        print("\n⚠️ Setup incomplete. Please check error messages above.")

if __name__ == "__main__":
    main()
