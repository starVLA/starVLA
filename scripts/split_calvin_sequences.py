#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def chunk_list(xs, n):
    k, m = divmod(len(xs), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        chunks.append(xs[start:start + size])
        start += size
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text())

    # Most eval_sequences.json files are simple list structures.
    # If it is a dict, try to split the first list-valued key while preserving other keys.
    if isinstance(data, list):
        seqs = data
        wrapper_type = "list"
        list_key = None
    elif isinstance(data, dict):
        list_key = None
        for k, v in data.items():
            if isinstance(v, list):
                list_key = k
                seqs = v
                break
        if list_key is None:
            raise ValueError("Input JSON is a dict but no list-valued key was found.")
        wrapper_type = "dict"
    else:
        raise ValueError(f"Unsupported JSON type: {type(data)}")

    if args.limit is not None:
        seqs = seqs[:args.limit]

    chunks = chunk_list(seqs, args.num_workers)

    manifest = []
    offset = 0
    for i, chunk in enumerate(chunks):
        if wrapper_type == "list":
            out_data = chunk
        else:
            out_data = dict(data)
            out_data[list_key] = chunk

        out_path = out_dir / f"eval_sequences_worker_{i}.json"
        count_path = out_dir / f"eval_sequences_worker_{i}.count"
        offset_path = out_dir / f"eval_sequences_worker_{i}.offset"

        out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
        count_path.write_text(f"{len(chunk)}\n")
        offset_path.write_text(f"{offset}\n")

        manifest.append(
            {
                "worker": i,
                "path": str(out_path),
                "count": len(chunk),
                "offset": offset,
            }
        )
        offset += len(chunk)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"Input sequences: {len(seqs)}")
    print(f"Workers: {args.num_workers}")
    print(f"Output dir: {out_dir}")
    for item in manifest:
        print(f"worker {item['worker']}: {item['count']} sequences -> {item['path']}")


if __name__ == "__main__":
    main()
