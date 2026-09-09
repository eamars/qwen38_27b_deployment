"""Bounded single-request MTP5 probe using the official 0731 draft sidecar."""
import json
import os
from pathlib import Path
import types
import torch
import torch.nn.functional as F
from .weight import _ShardReader, _weight_map, _iter_dspark_weights
from .args import dspark_enabled
_draft_banks = {}

def configure(args):
    folder=os.environ.get('FREETOKEN_MTP_SIDECAR')
    if not folder: return
    data=json.loads((Path(folder)/'config.json').read_text())
    for key in ('dim','n_layers','n_routed_experts','moe_inter_dim','vocab_size'):
        assert getattr(args,key)==data[key],key
    for key in ('n_mtp_layers','dspark_block_size','dspark_markov_rank','dspark_noise_token_id','dspark_target_layer_ids'):
        setattr(args,key,data[key])
    assert args.dspark_block_size==5

def weights(args):
    folder=os.environ.get('FREETOKEN_MTP_SIDECAR')
    if not folder or not dspark_enabled(): return
    configure(args); args.dspark_enabled=True
    reader=_ShardReader(folder,_weight_map(folder),'cpu')
    def linear(prefix,split=None):
        yield prefix+'.weight',reader.get(prefix+'.weight')
        if reader.has(prefix+'.scale'): yield prefix+'.scale',reader.get(prefix+'.scale')
    try: yield from _iter_dspark_weights(args,reader,linear)
    finally: reader.close()

def append_banks(banks,config):
    folder=os.environ.get('FREETOKEN_MTP_SIDECAR')
    if not folder or not config.dsv4_args.dspark_enabled: return
    from freetoken.moe.host_banks import alloc_layer_banks
    specs={'gate_up':((256,4096,1568),torch.uint8),'down':((256,4096,784),torch.uint8)}
    hb=alloc_layer_banks(specs,3)
    for k in range(3):
        for name,(shape,_) in specs.items():
            source=torch.from_file(str(Path(folder)/f'{name}-{k}.bin'),shared=False,size=256*shape[1]*shape[2],dtype=torch.uint8)
            hb[name][k].tensor.copy_(source.reshape(shape)); del source
            hb[name][k].lock()
            assert hb[name][k]._locked, 'Draft weights must remain locked in RAM'
            _draft_banks[(config.num_layers+k,name)]=hb[name][k]
            # These cache entries are never used: draft routed_forward below owns
            # RAM staging. Aliases preserve the engine's layer-index contract.
            banks[name].append(banks[name][0])
    print('MTP5: 6.89 GiB draft experts OS-locked in RAM; GPU-only staged compute',flush=True)

def install(drafter):
    if drafter is None or not os.environ.get('FREETOKEN_MTP_SIDECAR'): return
    drafter.store_context_kv=types.MethodType(store_context,drafter)
    drafter.propose=types.MethodType(propose,drafter)
    for block in drafter.layers:
        block.attn.forward_ragged=types.MethodType(attention,block.attn)
        block.ffn.experts.routed_forward=types.MethodType(routed,block.ffn.experts)

def routed(self,x,weights,ids):
    from freetoken.moe.fused_q4_0 import fused_experts_gguf
    unique,inverse=torch.unique(ids.cpu().long(),sorted=True,return_inverse=True)
    views=[]
    for name in ('gate_up','down'):
        host=_draft_banks[(self.layer_id,name)].tensor.index_select(0,unique)
        views.append(host.to(x.device))
    return fused_experts_gguf(x,*views,weights,inverse.to(x.device,dtype=ids.dtype),
                             'silu',quant_type=18,down_quant_type=18,swiglu_limit=self.swiglu_limit)

def store_context(self,main_x,positions,window_slots):
    from .ops import apply_rotary_emb_decode
    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
    # Only the live 128-token window is needed; absolute positions survive rollback.
    positions=positions[-128:].long(); main_x=main_x[-128:]
    for block in self.layers:
        a=block.attn
        if not hasattr(a,'_mtp_kv'):
            a._mtp_kv=torch.zeros((128,a.head_dim),dtype=main_x.dtype,device=main_x.device)
            a._mtp_pos=torch.full((128,),-1,dtype=torch.long,device=main_x.device)
        kv=a.kv_norm(a.wkv(main_x))
        apply_rotary_emb_decode(kv[:,-a.rope_head_dim:].unsqueeze(1),a.freqs_cis.index_select(0,positions))
        act_quant_fp8_inplace(kv[:,:-a.rope_head_dim],64)
        a._mtp_kv[positions%128]=kv; a._mtp_pos[positions%128]=positions

def attention(self,x,segments,positions):
    from .ops import apply_rotary_emb
    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
    assert len(segments)==1 and x.shape[1]==5
    rd=self.rope_head_dim; n=x.shape[1]
    freqs=self.freqs_cis.index_select(0,positions.long())
    q=self.wq_b(self.q_norm(self.wq_a(x))).unflatten(-1,(self.n_heads,self.head_dim))
    q=q*torch.rsqrt(q.square().mean(-1,keepdim=True)+self.eps)
    apply_rotary_emb(q[...,-rd:],freqs)
    kv=self.kv_norm(self.wkv(x)); apply_rotary_emb(kv[...,-rd:],freqs)
    act_quant_fp8_inplace(kv[...,:-rd],64)
    bank=torch.cat([self._mtp_kv,kv[0]],dim=0)
    valid=(self._mtp_pos>=positions[0]-128)&(self._mtp_pos<positions[0])
    valid=torch.cat([valid,torch.ones(n,dtype=torch.bool,device=x.device)])
    scores=torch.einsum('bthd,sd->bhts',q.float(),bank.float())*self.softmax_scale
    scores=scores.masked_fill(~valid,float('-inf'))
    sink=self.attn_sink.view(1,-1,1,1).expand(1,-1,n,1)
    prob=torch.softmax(torch.cat([scores,sink],dim=-1),dim=-1)[...,:-1]
    out=torch.einsum('bhts,sd->bthd',prob,bank.float()).to(x.dtype)
    apply_rotary_emb(out[...,-rd:],freqs,True)
    return self._wo(out,1,n)

def propose(self,aux,input_ids,segments,positions,target_logits_fn):
    assert len(segments)==1 and input_ids.numel()==6
    ids=input_ids[:,:5]; pos=positions[:5]
    off,_,ti,start=segments[0]
    hidden=self.head_hidden(self.embed_block(ids),ids,[(off,5,ti,start)],pos)[0]
    base=target_logits_fn(hidden)
    previous=ids[0,0].view(1)
    logits=[]
    for i in range(5):
        row=base[i:i+1]+self.markov_head.bias(self.markov_head.embed(previous))
        logits.append(row); previous=row.argmax(-1)
    # The existing verifier consumes k+1 rows and drops the final proposal.
    return torch.cat(logits+[logits[-1]],dim=0),torch.ones(6,device=hidden.device)
