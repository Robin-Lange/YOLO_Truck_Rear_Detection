import torch
import ultralytics


def sanity_check():
    print(f"--- HARDWARE CHECK ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA Active: Yes")
        print(f"✅ GPU Found: {gpu_name} (Count: {gpu_count})")

        # Memory Check
        vram_free, vram_total = torch.cuda.mem_get_info()
        print(
            f"✅ VRAM Status: {vram_free/1e9:.2f}GB free / {vram_total/1e9:.2f}GB total"
        )
    else:
        print(
            "❌ CRITICAL: CUDA not available. Training will be on CPU (Years to finish)."
        )
        exit()

    print(f"\n--- SOFTWARE ACCELERATION ---")
    # Verify FP16 (Half Precision) Capability - Critical for Tensor Cores
    try:
        x = torch.tensor([1.0], device="cuda", dtype=torch.float16)
        print("✅ FP16 (AMP) Support: Active")
    except Exception as e:
        print(f"❌ FP16 Error: {e}")

    # Verify YOLO Version
    print(f"✅ YOLO Version: {ultralytics.__version__}")

    # CuDNN Benchmark (Optimizes convolution algorithms for your specific input size)
    if torch.backends.cudnn.enabled:
        print("✅ cuDNN Enabled: Yes")
        torch.backends.cudnn.benchmark = True
        print("✅ cuDNN Benchmark: Set to True (Max Speed)")


if __name__ == "__main__":
    sanity_check()
