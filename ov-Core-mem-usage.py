#!/usr/bin/env python3
import psutil
import openvino as ov
import numpy as np
import gc
from memory_profiler import profile


def bytes_state():
    this_process = psutil.Process()
    return psutil.virtual_memory().total - psutil.virtual_memory().available, sum((proc.memory_info().rss for proc in this_process.children(recursive=True)), start=this_process.memory_info().rss)


def print_memory_usage(old, new):
    # print(f"System memory used: {(new[0] - old[0]) / (1024 ** 2)} MB")
    print(f"Process memory used: {(new[1] - old[1]) / (1024 ** 2)} MB")


def n_models_usage(cores, device):
    model_device = "NPU" if device == "NPU:CPU" else device
    old = bytes_state()
    holder = []
    for core in cores:
        param1 = ov.opset8.parameter([], np.float32)
        compiled_model = core.compile_model(ov.Model([param1], [param1], "identity"), model_device, {'NPU_USE_NPUW': 'YES', 'NPUW_DEVICES': 'CPU', 'NPUW_ONLINE_PIPELINE': 'NONE'} if device == "NPU:CPU" else {})
        ireq = compiled_model.create_infer_request()
        ireq.infer()
        holder.append(ireq)
    new = bytes_state()
    return (new[1] - old[1]) / (1024 ** 2)


@profile
def list_usage():
    """Verify psutil and memory_profiler are aligned"""
    old = bytes_state()
    holder = []
    for i in range(1_000_000):
        holder.append(i)
    new = bytes_state()
    print_memory_usage(old, new)


def main():
    holder = list_usage()
    holder2 = list_usage()
    del holder
    del holder2
    gc.collect()
    for device in "MULTI:CPU", "NPU:CPU", "CPU", "GPU", "NPU":
        n_models_usage([ov.Core()], device)
    print("Warmed up\n")
    for device in "MULTI:CPU", "NPU:CPU", "CPU", "GPU", "NPU":
        for n in 1, 2, 4, 8, 16, 32, 1000:
            gc.collect()
            n_models_usage((ov.Core() for _ in range(n)), device)  # Warm up
            gc.collect()
            mb_used_1 = n_models_usage((ov.Core() for _ in range(n)), device)
            gc.collect()
            mb_used_2 = n_models_usage((ov.Core() for _ in range(n)), device)
            gc.collect()
            mb_used_3 = n_models_usage((ov.Core() for _ in range(n)), device)
            gc.collect()
            core = ov.Core()
            mb_used_4 = n_models_usage((core for _ in range(n)), device)
            del core
            gc.collect()
            core = ov.Core()
            mb_used_5 = n_models_usage((core for _ in range(n)), device)
            del core
            gc.collect()
            core = ov.Core()
            mb_used_6 = n_models_usage((core for _ in range(n)), device)
            del core
            print(f"{device} {n} Models: {mb_used_1:8.2f} MB | {mb_used_2:8.2f} MB | {mb_used_3:8.2f} MB || {mb_used_4:8.2f} MB | {mb_used_5:8.2f} MB | {mb_used_6:8.2f} MB")
            print("Average: {:.2f} MB | {:.2f} MB".format(
                (mb_used_1 + mb_used_2 + mb_used_3) / 3,
                (mb_used_4 + mb_used_5 + mb_used_6) / 3,
            ))
            print()


if __name__ == "__main__":
    main()
