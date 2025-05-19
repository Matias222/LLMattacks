import torch

def check_cuda():
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        device_id = torch.cuda.current_device()
        print(f"GPU Name: {torch.cuda.get_device_name(device_id)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")

        # Get memory usage info
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3

        print(f"\n📊 VRAM Status:")
        print(f"  Total      : {total:.2f} GB")
        print(f"  Reserved   : {reserved:.2f} GB")
        print(f"  Allocated  : {allocated:.2f} GB")
        print(f"  Free (est) : {reserved - allocated:.2f} GB")
    else:
        print("❌ CUDA is NOT available.")

if __name__ == "__main__":
    check_cuda()
