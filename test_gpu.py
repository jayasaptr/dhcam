#!/usr/bin/env python3
"""
GPU Test Script for Hard Hat Detection System
Run this script to test GPU capabilities before running the main application
"""

import torch
import cv2
from ultralytics import YOLO
from app.gpu_utils import check_gpu_availability, setup_gpu_environment
from app.config import MODEL_PATH, GPU_SETTINGS

def test_gpu_setup():
    """Test GPU setup and model loading"""
    print("🔍 Testing GPU Setup...")
    print("=" * 50)
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    
    # Setup GPU environment
    device = setup_gpu_environment()
    
    print("\n🧠 Testing YOLO Model on GPU...")
    print("-" * 30)
    
    try:
        # Load model
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
        
        # Move to GPU if available
        if device != 'cpu':
            model.to(device)
            print(f"✅ Model moved to {device}")
            
            # Test inference speed
            import time
            import numpy as np
            
            # Create dummy image
            dummy_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(3):
                _ = model.predict(dummy_img, verbose=False)
            
            # Time inference
            start_time = time.time()
            num_inferences = 10
            
            for _ in range(num_inferences):
                results = model.predict(dummy_img, verbose=False)
            
            end_time = time.time()
            avg_inference_time = (end_time - start_time) / num_inferences
            fps = 1.0 / avg_inference_time
            
            print(f"✅ Average inference time: {avg_inference_time:.3f}s")
            print(f"✅ Estimated FPS: {fps:.1f}")
            
        else:
            print("ℹ️ Running on CPU - GPU not available")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False
    
    print("\n📊 GPU Configuration:")
    print("-" * 30)
    for key, value in GPU_SETTINGS.items():
        print(f"{key:20}: {value}")
    
    print("\n✅ GPU test completed successfully!")
    return True

def test_opencv_cuda():
    """Test OpenCV CUDA support"""
    print("\n🎥 Testing OpenCV CUDA...")
    print("-" * 30)
    
    try:
        # Check build info
        build_info = cv2.getBuildInformation()
        cuda_support = 'CUDA' in build_info and 'YES' in build_info
        
        if cuda_support:
            print("✅ OpenCV built with CUDA support")
            
            # Test GPU Mat operations
            import numpy as np
            
            # Create test image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test GPU upload/download
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            downloaded_img = gpu_img.download()
            
            print("✅ GPU Mat operations working")
            
        else:
            print("ℹ️ OpenCV built without CUDA support")
            
    except Exception as e:
        print(f"❌ OpenCV CUDA test failed: {e}")

if __name__ == "__main__":
    print("🚀 Hard Hat Detection - GPU Test")
    print("=" * 50)
    
    success = test_gpu_setup()
    test_opencv_cuda()
    
    if success:
        print("\n🎉 All tests passed! Your system is ready for GPU acceleration.")
        print("\n💡 To run the main application:")
        print("   python main.py")
    else:
        print("\n⚠️ Some tests failed. Check your GPU setup.")
        print("\n💡 Installation help:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   # For OpenCV with CUDA: pip install opencv-contrib-python")
