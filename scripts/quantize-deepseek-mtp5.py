"""Convert only official draft experts to the target's IQ3_XXS bank format."""
import ctypes as C
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from safetensors import safe_open
import struct

root=Path('/home/rba90/deepseek-freetoken-probe/mtp5')
lib=C.CDLL('/home/rba90/deepseek-freetoken-probe/ggml-quant/build/src/libggml-base.so')
lib.ggml_quantize_chunk.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int64,C.c_int64,C.c_int64,C.c_void_p]
lib.ggml_quantize_chunk.restype=C.c_size_t
lib.ggml_quantize_init(18)
mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
lut=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
headers={}
for shard in sorted(set(v for k,v in mapping.items() if k.startswith('mtp.'))):
    with (root/shard).open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        headers[shard]=(8+size,json.loads(f.read(size)))
def raw(name):
    shard=mapping[name]; offset,header=headers[shard]; t=header[name]
    with (root/shard).open('rb') as f:
        a,b=t['data_offsets']; f.seek(offset+a)
        return np.frombuffer(f.read(b-a),np.uint8).reshape(t['shape'])
def quant(k,e,proj):
    name=f'mtp.{k}.ffn.experts.{e}.{proj}'
    w=raw(name+'.weight'); s=raw(name+'.scale')
    # safetensors numpy cannot expose float8 scales; read raw tensor bytes below if needed.
    n,packed=w.shape
    x=np.empty((n,packed*2),np.float32)
    x[:,0::2]=lut[w & 15]; x[:,1::2]=lut[w >> 4]
    scale=np.exp2(s.view(np.uint8).astype(np.float32)-127)
    x.reshape(n,-1,32)[:] *= scale.reshape(n,-1,1)
    out=np.empty((n,packed*2//256*98),np.uint8)
    got=lib.ggml_quantize_chunk(18,x.ctypes.data,out.ctypes.data,0,n,packed*2,None)
    assert got==out.nbytes
    return e,proj,out
started=time.time()
for k in range(3):
    done=root/f'iq3-layer-{k}.done'
    if done.exists(): continue
    gu=np.memmap(root/f'gate_up-{k}.bin',mode='w+',dtype=np.uint8,shape=(256,4096,1568))
    down=np.memmap(root/f'down-{k}.bin',mode='w+',dtype=np.uint8,shape=(256,4096,784))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for base in range(0,256,8):
            futures=[pool.submit(quant,k,e,p) for e in range(base,base+8) for p in ('w1','w3','w2')]
            for f in futures:
                e,p,out=f.result()
                if p=='w2': down[e]=out
                if p=='w1': gu[e,:2048]=out
                if p=='w3': gu[e,2048:]=out
            del futures,out
            print(f'layer={k} experts={base+8}/256 elapsed={time.time()-started:.1f}s',flush=True)
    gu.flush(); down.flush(); del gu,down
    done.write_text('official 0731 -> IQ3_XXS, ggml 7840aab\n')
print('Conversion complete',flush=True)
