import subprocess

def get_nvidia_gpu_name():
    try:
        # Execute the 'nvidia-smi' command and capture its output
        gpu_info = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    except Exception as e:
        # Handle the case where 'nvidia-smi' is not found
        print(f"[INFO] Error: {e}, not connected to a NVIDIA GPU, setting GPU_NAME to None")
        return None

    # If the command was successful, parse the GPU name
    try:
        # Execute 'nvidia-smi -L' command to get detailed GPU info
        gpu_full_name = subprocess.check_output(['nvidia-smi', '-L'], encoding='utf-8')
        gpu_name = gpu_full_name.split(":")[1].split("(")[0].strip()
        print(f"[INFO] Connected to NVIDIA GPU: {gpu_name}")
        return gpu_name
    except Exception as e:
        print(f"Error occurred while getting GPU name: {e}")
        return None