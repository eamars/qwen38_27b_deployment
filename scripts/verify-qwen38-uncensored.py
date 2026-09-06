"""Validate local checkpoint headers against the isolated loader; no tensor payload reads."""

import argparse
import collections
import json
import math
from pathlib import Path
import re
import struct

EXPERT = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale|weight_global_scale)$")
PLE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8,
         "I32": 4, "U8": 1, "F8_E4M3": 1, "BOOL": 1}


def verify(folder):
    manifest = json.loads((folder / "staging-manifest.json").read_text())
    assert manifest["complete"], "Checkpoint staging is incomplete"
    assert manifest["revision"] == "c1209bda15a6bbc4c68b585e93d40c0d85f50306"
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((folder / "config.json").read_text())["text_config"]
    hidden, intermediate = config["hidden_size"], config["moe_intermediate_size"]
    layers, experts = config["num_hidden_layers"], config["num_experts"]
    records = {e["name"]: e for e in manifest["files"]}
    counts = collections.Counter()
    seen, dense, ple = set(), {}, {}
    header_bytes = 0
    for filename in sorted(set(index.values())):
        path = folder / filename
        entry = records[filename]
        assert entry["verified"] and path.stat().st_size == entry["size"], filename
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            assert 0 < length <= 64 << 20, filename
            metadata = json.loads(handle.read(length))
        header_bytes += length + 8
        for key, tensor in metadata.items():
            if key == "__metadata__":
                continue
            assert key not in seen and index.get(key) == filename, key
            seen.add(key)
            begin, end = tensor["data_offsets"]
            assert 0 <= begin <= end <= entry["size"] - length - 8, key
            assert end - begin == math.prod(tensor["shape"]) * BYTES[tensor["dtype"]], key
            if key.startswith(("mtp.", "model.visual.", "visual.")):
                counts["unused_tensors"] += 1
                continue
            match = EXPERT.match(key)
            if match:
                layer, expert, projection, kind = match.groups()
                assert 0 <= int(layer) < layers and 0 <= int(expert) < experts, key
                rows, cols = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
                expected = {"weight_packed": ("U8", [rows, cols // 2]),
                            "weight_scale": ("F8_E4M3", [rows, cols // 16]),
                            "weight_global_scale": ("F32", [1])}[kind]
                assert (tensor["dtype"], tensor["shape"]) == expected, key
                counts[kind] += 1
            elif PLE.search(key):
                shard = int(PLE.search(key).group(1))
                assert shard not in ple and tensor["dtype"] == "BF16", key
                assert tensor["shape"][1] == 160, key
                ple[shard] = tensor["shape"][0]
            else:
                assert ".mlp.experts." not in key, f"Unrecognized expert tensor: {key}"
                dense[key] = tensor
    assert seen == set(index), "Header/index tensor set mismatch"
    expected_projections = layers * experts * 3
    for kind in ("weight_packed", "weight_scale", "weight_global_scale"):
        assert counts[kind] == expected_projections, (kind, counts[kind])
    assert sorted(ple) == list(range(config["split_ngram_parts"])), "Incomplete PLE table"
    assert len(set(ple.values())) == 1, "PLE extents have unequal row counts"
    for key, tensor in dense.items():
        if tensor["dtype"] == "F8_E4M3":
            assert key.endswith(".weight"), key
            scale = dense[key.removesuffix(".weight") + ".weight_scale"]
            assert scale["shape"] in ([tensor["shape"][0], 1], [1], []), key
            assert scale["dtype"] in ("BF16", "F16", "F32"), key
            counts["fp8_dense_weights"] += 1
    return {"passed": True, "revision": manifest["revision"],
            "files_verified_by_downloader": len(records), "indexed_tensors": len(seen),
            "header_bytes_read": header_bytes, "counts": dict(counts),
            "ple_dtype": "BF16", "ple_extents": len(ple), "ple_rows": sum(ple.values()),
            "model_initialized": False, "tensor_payloads_read_by_this_check": False,
            "live_inference_tested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(
        "/home/rba90/models/Qwen3.8-Flash-Next-Uncensored-NVFP4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.model_dir)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
