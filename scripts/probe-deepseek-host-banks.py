"""Exercise FreeToken's exact host-bank allocation and registration path."""
import json
import pathlib
import time
import torch
from freetoken.moe.host_banks import HostBank
from freetoken.kernel.pinned import device_ptr

torch.set_num_threads(8)
held = []
result = {'target_layers': 43, 'completed_layers': 0, 'passed': False}
start = time.perf_counter()
try:
    for layer in range(43):
        for shape in [(256,4096,1568),(256,4096,784)]:
            available = int(next(x for x in pathlib.Path('/proc/meminfo').read_text().splitlines()
                                 if x.startswith('MemAvailable:')).split()[1])*1024
            size = shape[0]*shape[1]*shape[2]
            if available-size < 3*2**30:
                raise RuntimeError('Preserving 3 GiB available RAM')
            bank = HostBank(shape,torch.uint8)
            bank.tensor.fill_(layer)
            bank.pin()
            assert device_ptr(bank.tensor) != 0
            held.append(bank)
        result['completed_layers'] = layer+1
        if (layer+1)%10 == 0:
            print(json.dumps({'completed_layers':layer+1}),flush=True)
    result['passed'] = True
except Exception as exc:
    result['error'] = repr(exc)
result['allocated_gib'] = sum(b.nbytes for b in held)/2**30
result['elapsed_s'] = time.perf_counter()-start
print(json.dumps(result),flush=True)
pathlib.Path('/home/rba90/deepseek-freetoken-probe/host-bank-probe.json').write_text(json.dumps(result,indent=2))
