"""Exercise the shared Qwen4 loaders with tiny synthetic files and no CUDA initialization."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/freetoken-eamars/python"))

import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info
from freetoken.kernel import _ple_store
from freetoken.layers.quantization import NameMap, QuantConfig, set_quant_config
from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces
from freetoken.models.qwen4_exp import config, ple_disk, weight
from freetoken.models.register import get_model_spec
from freetoken.utils import cached_load_hf_config


def install_quant_config(model: Path):
    hf_config = cached_load_hf_config(str(model))
    spec = get_model_spec(hf_config.architectures[0])
    quant = QuantConfig.from_hf(
        hf_config,
        name_map=NameMap(
            roots=spec.checkpoint_roots,
            segments=spec.checkpoint_segments,
            packed=spec.packed_modules_mapping,
        ),
        unquantized=spec.unquantized_modules,
    )
    set_quant_config(quant)
    return hf_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    set_tp_info(0, 1)
    checks = []
    model = Path("/home/rba90/models/Qwen3.8-Flash-Next-Uncensored-NVFP4")
    hf_config = install_quant_config(model)
    parsed = config.parse_config(hf_config)
    assert parsed.expert_quant == "nvfp4"
    checks.append("real checkpoint config selects compressed-tensors NVFP4 expert banks")

    with tempfile.TemporaryDirectory(prefix="qwen4-shared-loader-probe-") as temp:
        folder = Path(temp)
        (folder / "config.json").write_bytes((model / "config.json").read_bytes())
        prefix = "model.language_model.layers.0"
        dense, scales, experts = {}, {}, {}
        expected_qkvz, expected_qkvz_scale = [], []
        for i, projection in enumerate(("in_proj_qkv", "in_proj_z"), 1):
            name = f"{prefix}.linear_attn.{projection}"
            value = (torch.arange(-64, 0).reshape(2, 32) + i).to(torch.float8_e4m3fn)
            scale = torch.tensor([[i / 8], [i / 4]], dtype=torch.bfloat16)
            dense[name + ".weight"] = value
            scales[name + ".weight_scale"] = scale
            expected_qkvz.append(value)
            expected_qkvz_scale.append(scale.reshape(-1))

        dense[f"{prefix}.linear_attn.in_proj_b.weight"] = torch.ones((2, 32), dtype=torch.bfloat16)
        dense[f"{prefix}.linear_attn.in_proj_a.weight"] = torch.full((2, 32), 2, dtype=torch.bfloat16)

        for i, projection in enumerate(("gate_proj", "up_proj"), 1):
            name = f"{prefix}.mlp.shared_expert.{projection}"
            value = (torch.arange(-8, 0).reshape(2, 4) + i).to(torch.float8_e4m3fn)
            scale = torch.tensor([[i / 8], [i / 4]], dtype=torch.bfloat16)
            dense[name + ".weight"] = value
            scales[name + ".weight_scale"] = scale

        dense[f"{prefix}.attn_hyper_connection.input_mix_weight_down.weight"] = torch.ones((3, 4), dtype=torch.bfloat16)
        dense[f"{prefix}.attn_hyper_connection.block_inject_weight.weight"] = torch.full((2, 4), 2, dtype=torch.bfloat16)
        dense["mtp.test.weight"] = torch.ones(1)

        for projection, value in (("gate_proj", 1), ("up_proj", 2), ("down_proj", 3)):
            name = f"{prefix}.mlp.experts.0.{projection}"
            rows, cols = (16, 16)
            experts[name + ".weight_packed"] = torch.full((rows, cols // 2), value, dtype=torch.uint8)
            experts[name + ".weight_scale"] = torch.ones((rows, cols // 16), dtype=torch.float8_e4m3fn)
            experts[name + ".weight_global_scale"] = torch.tensor([float(value * 2)], dtype=torch.float32)

        save_file(dense, str(folder / "dense.safetensors"))
        save_file(scales, str(folder / "scales.safetensors"))
        save_file(experts, str(folder / "experts.safetensors"))
        index = {name: "dense.safetensors" for name in dense}
        index.update({name: "scales.safetensors" for name in scales})
        index.update({name: "experts.safetensors" for name in experts})
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))

        dense_loaded = dict(weight.iter_weights(
            str(folder), torch.device("cpu"), include_moe_experts=False, include_non_moe=True,
        ))
        assert "mtp.test.weight" not in dense_loaded
        assert set(dense_loaded) == {
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            "model.layers.0.linear_attn.in_proj_qkvz.weight_scale",
            "model.layers.0.linear_attn.in_proj_ba.weight",
            "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
            "model.layers.0.mlp.shared_expert.gate_up_proj.weight_scale",
            "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        }
        torch.testing.assert_close(
            dense_loaded["model.layers.0.linear_attn.in_proj_qkvz.weight"],
            torch.cat(expected_qkvz), rtol=0, atol=0,
        )
        torch.testing.assert_close(
            dense_loaded["model.layers.0.linear_attn.in_proj_qkvz.weight_scale"],
            torch.cat(expected_qkvz_scale), rtol=0, atol=0,
        )
        checks.append("FP8 channel scales and Qwen4 packed projections load through the shared reader")

        tiny = SimpleNamespace(num_experts=1, num_layers=1, num_moe_layers=1)
        spec = weight.nvfp4_expert_spec(str(folder), tiny)
        pieces = list(iter_nvfp4_expert_pieces(
            str(folder), tiny, spec, primary=True, drop_page_cache=lambda _: None,
        ))
        assert len(pieces) == 1
        _, e0, e1, piece = pieces[0]
        assert (e0, e1) == (0, 1)
        assert torch.all(piece["gate"][0] == 1)
        assert torch.all(piece["up"][0] == 2)
        assert torch.all(piece["down"][0] == 3)
        torch.testing.assert_close(piece["gate_global"], torch.full_like(piece["gate_global"], 0.5))
        torch.testing.assert_close(piece["up_global"], torch.full_like(piece["up_global"], 0.25))
        torch.testing.assert_close(piece["down_global"], torch.full_like(piece["down_global"], 1 / 6))
        checks.append("compressed-tensors expert names and reciprocal global scales load through the bank reader")

        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            table_folder = folder / str(dtype).split(".")[-1]
            table_folder.mkdir()
            table = (torch.arange(16 * 160).reshape(16, 160) % 32).to(dtype)
            ple_name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
            table_tensors = {
                f"{ple_name}.shard_{i}.weight": table[i * 8:(i + 1) * 8].clone()
                for i in range(2)
            }
            if dtype == torch.float8_e4m3fn:
                table_tensors[ple_name + ".weight_scale"] = torch.tensor(.5)
            save_file(table_tensors, str(table_folder / "table.safetensors"))
            (table_folder / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {name: "table.safetensors" for name in table_tensors}}
            ))
            source = ple_disk.source_from_safetensors(str(table_folder))
            assert source.storage_dtype == dtype and source.row_bytes == 160 * dtype.itemsize
            tokens = torch.tensor([0, 0, 6, 7, 8], dtype=torch.int64)
            staging = torch.empty(3 * 2 * source.row_bytes, dtype=torch.uint8)
            store = _ple_store.PleStore(
                source.paths, source.extent_file, source.extent_base,
                source.rows_per_extent, source.row_bytes, source.row_stride,
                [1, 0, 0], [8, 8], [0, 8], -1, True,
            )
            store.stage(tokens.data_ptr(), 3, staging.data_ptr())
            store.flush()
            actual = staging.view(source.storage_dtype).to(torch.bfloat16).reshape(6, 160) * source.scale
            expected = table.to(torch.bfloat16)[[6, 14, 7, 15, 0, 8]] * source.scale
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checks.append(f"compiled PLE store reads {dtype} rows with the correct byte stride")
            del store

    assert not torch.cuda.is_initialized()
    report = {
        "passed": True,
        "checks": checks,
        "cuda_initialized": False,
        "real_model_weights_loaded": False,
        "live_inference_tested": False,
        "runtime_source": str(Path(weight.__file__).resolve()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
