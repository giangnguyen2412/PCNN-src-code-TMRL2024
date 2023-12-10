import torch
import multiprocessing

# Set the GPUs to use
gpu_ids = [6,7]

# Define a function to allocate memory on the specified GPU
def allocate_memory(gpu_id):
    torch.cuda.set_device(gpu_id)
    # Allocate a large tensor once
    tensor = torch.randn((100000, 40000), device=f"cuda:{gpu_id}")
    # Keep the process alive
    while True:
        torch.cuda.synchronize()

# Start a process to allocate memory on each GPU
processes = []
for gpu_id in gpu_ids:
    process = multiprocessing.Process(target=allocate_memory, args=(gpu_id,))
    process.start()
    processes.append(process)

# Wait for all processes to finish (which will never happen, since they run indefinitely)
for process in processes:
    process.join()
