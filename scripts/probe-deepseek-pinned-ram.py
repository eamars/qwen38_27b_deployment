"""Bounded CUDA-pinned allocation probe for the 98.77 GiB expert bank."""
import json
import pathlib
import time
import torch

def memory():
    return {line.split(':')[0]: int(line.split()[1])*1024
            for line in pathlib.Path('/proc/meminfo').read_text().splitlines()}

torch.set_num_threads(8)
blocks = []
result = {'requested_gib': 99, 'start': memory(), 'allocated_gib': 0}
t = time.perf_counter()
try:
    for i in range(99):
        if memory()['MemAvailable'] < 4*2**30:
            raise RuntimeError('Stopped to preserve at least 3 GiB available RAM')
        block = torch.empty(2**30, dtype=torch.uint8, pin_memory=True)
        block.fill_(i)
        blocks.append(block)
        result['allocated_gib'] = len(blocks)
        if len(blocks) % 10 == 0:
            print(json.dumps({'allocated_gib':len(blocks), 'available_gib':memory()['MemAvailable']/2**30}),flush=True)
    result['passed'] = True
except Exception as e:
    result['passed'] = False
    result['error'] = repr(e)
result['peak'] = memory()
result['elapsed_s'] = time.perf_counter()-t
print(json.dumps(result),flush=True)
pathlib.Path('/home/rba90/deepseek-freetoken-probe/pinned-ram-probe.json').write_text(json.dumps(result,indent=2))
