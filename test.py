import torch
import multiprocessing

# set the GPUs to use
gpu_ids = [0,1]

# define a function to allocate memory on the specified GPU
def allocate_memory(gpu_id):
    torch.cuda.set_device(gpu_id)
    while True:
        tensor = torch.randn((100000, 20000), device=f"cuda:{gpu_id}")
        torch.cuda.empty_cache()

# start a process to allocate memory on each GPU
processes = []
for gpu_id in gpu_ids:
    process = multiprocessing.Process(target=allocate_memory, args=(gpu_id,))
    process.start()
    processes.append(process)

# wait for all processes to finish (which will never happen, since they run indefinitely)
for process in processes:
    process.join()
