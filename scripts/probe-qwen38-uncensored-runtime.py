"""Exercise the isolated loaders with tiny synthetic files and no CUDA initialization."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/freetoken-qwen38-uncensored/python"))

import torch
from safetensors.torch import save_file
from freetoken.distributed import set_tp_info
from freetoken.kernel import _ple_store
from freetoken.models.qwen4_exp import config, weight, ple_disk
from freetoken.utils import cached_load_hf_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    set_tp_info(0, 1)
    checks = []
    model = Path("/home/rba90/models/Qwen3.8-Flash-Next-Uncensored-NVFP4")
    parsed = config.parse_config(cached_load_hf_config(str(model)))
    assert (parsed.expert_quant, parsed.attn_quant, parsed.dense_quant, parsed.lm_head_quant) == (
        "nvfp4", "none", "none", "none")
    checks.append("real checkpoint config selects NVFP4 expert banks and BF16 dense buffers")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    for thinking in (True, False):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with the number 7."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        rendered = tokenizer.decode(ids)
        assert "Reply with the number 7." in rendered and "<|im_start|>assistant" in rendered
        checks.append(f"local tokenizer renders thinking={thinking} chat template ({len(ids)} tokens)")

    with tempfile.TemporaryDirectory(prefix="uncensored-loader-probe-") as temp:
        folder = Path(temp)
        (folder / "config.json").write_bytes((model / "config.json").read_bytes())
        prefix = "model.language_model.layers.0"
        tensors, scales = {}, {}
        expected_qkv = []
        for i, projection in enumerate(("q_proj", "k_proj", "v_proj"), 1):
            name = f"{prefix}.self_attn.{projection}"
            value = torch.arange(-32, 32).reshape(2, 32).to(torch.float8_e4m3fn)
            scale = torch.tensor([[i / 8], [i / 4]], dtype=torch.bfloat16)
            tensors[name + ".weight"] = value
            scales[name + ".weight_scale"] = scale
            expected_qkv.append((value.float() * scale.float()).to(torch.bfloat16))
        for i, projection in enumerate(("gate_proj", "up_proj", "down_proj"), 1):
            name = f"{prefix}.mlp.experts.0.{projection}"
            rows, cols = (32, 16) if projection == "down_proj" else (16, 32)
            tensors[name + ".weight_packed"] = torch.full((rows, cols // 2), i, dtype=torch.uint8)
            tensors[name + ".weight_scale"] = torch.ones(rows, cols // 16).to(torch.float8_e4m3fn)
            scales[name + ".weight_global_scale"] = torch.tensor([float(i * 2)])
        tensors["mtp.test.weight"] = torch.ones(1)
        save_file(tensors, str(folder / "weights.safetensors"))
        save_file(scales, str(folder / "scales.safetensors"))
        index = {name: "weights.safetensors" for name in tensors}
        index.update({name: "scales.safetensors" for name in scales})
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
        dense = dict(weight.iter_weights(str(folder), torch.device("cpu"),
                                        include_moe_experts=False, include_non_moe=True))
        assert set(dense) == {"model.layers.0.self_attn.qkv_proj.weight"}
        torch.testing.assert_close(next(iter(dense.values())), torch.cat(expected_qkv), rtol=0, atol=0)
        checks.append("FP8 channel scales across files applied before QKV fusion; experts/MTP omitted")

        tiny = SimpleNamespace(num_experts=1, hidden_size=32, moe_intermediate_size=16,
                               num_layers=1, first_k_dense_replace=0)
        retained_banks = []
        for loader in (weight.load_nvfp4_expert_sources, weight.load_nvfp4_expert_sources_parallel):
            banks = loader(str(folder), tiny, layer_sink=lambda layer, bank: retained_banks.append(bank))
            assert torch.all(banks["gate_up_packed"][0][0, :16] == 1)
            assert torch.all(banks["gate_up_packed"][0][0, 16:] == 2)
            assert torch.all(banks["down_packed"][0] == 3)
            torch.testing.assert_close(banks["gate_up_global"][0][0, :16], torch.full((16,), .5, dtype=torch.float16))
            torch.testing.assert_close(banks["gate_up_global"][0][0, 16:], torch.full((16,), .25, dtype=torch.float16))
            torch.testing.assert_close(banks["down_global"][0], torch.full_like(banks["down_global"][0], 1 / 6))
        checks.append("serial and O_DIRECT parallel expert banks preserve packed bytes and invert global scales")

        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            table_folder = folder / str(dtype).split(".")[-1]
            table_folder.mkdir()
            table = (torch.arange(16 * 160).reshape(16, 160) % 32).to(dtype)
            ple_name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
            table_tensors = {f"{ple_name}.shard_{i}.weight": table[i * 8:(i + 1) * 8].clone() for i in range(2)}
            if dtype == torch.float8_e4m3fn:
                table_tensors[ple_name + ".weight_scale"] = torch.tensor(.5)
            save_file(table_tensors, str(table_folder / "table.safetensors"))
            (table_folder / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {name: "table.safetensors" for name in table_tensors}}))
            source = ple_disk.source_from_safetensors(str(table_folder))
            assert source.storage_dtype == dtype and source.row_bytes == 160 * dtype.itemsize
            tokens = torch.tensor([0, 0, 6, 7, 8], dtype=torch.int64)
            staging = torch.empty(3 * 2 * source.row_bytes, dtype=torch.uint8)
            store = _ple_store.PleStore(
                source.paths, source.extent_file, source.extent_base,
                source.rows_per_extent, source.row_bytes, source.row_stride,
                [1, 0, 0], [8, 8], [0, 8], -1, True)
            store.stage(tokens.data_ptr(), 3, staging.data_ptr())
            store.flush()
            actual = staging.view(source.storage_dtype).to(torch.bfloat16).reshape(6, 160) * source.scale
            expected = table.to(torch.bfloat16)[[6, 14, 7, 15, 0, 8]] * source.scale
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checks.append(f"compiled PLE store reads {dtype} rows across extents/pages using {store.io_backend()}")
            del store
    assert not torch.cuda.is_initialized()
    report = {"passed": True, "checks": checks, "cuda_initialized": False,
              "real_model_weights_loaded": False, "live_inference_tested": False,
              "runtime_source": str(Path(weight.__file__).resolve())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
