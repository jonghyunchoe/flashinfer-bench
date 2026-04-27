#!/usr/bin/env python3
"""
Workload collection with HuggingFace push.

Pipeline:
  1. Run bench_sharegpt.py once across all batch sizes against a single
     SGLang server. FLASHINFER_DUMP_DEDUP=1 caps dumps to ~2 per unique
     (batch_size, kv_length) shape signature, so a small DUMP_MAX_COUNT
     budget covers every batch size without needing a per-batch restart.
  2. sanitize_dumps.py once over the combined dump dir → diverse workloads
     across all batch sizes.
  3. Push JSONL + new blob safetensors to the HF PR.
  4. flashinfer-bench run --local (baseline eval) → push trace JSONL.

Usage:
    # Must be invoked under tools/gpu-lock so CUDA_VISIBLE_DEVICES is set.
    tools/gpu-lock --gpus 8 --exec-timeout 10800 -- \\
      python3 scripts/collect_stream.py \\
        --def-name  gqa_paged_decode_h5_kv1_d128_ps64 \\
        --model-key llama-4-scout-ps64 \\
        --model-path /path/to/model \\
        --batch-sizes 64 128 \\
        --pr-num 263 \\
        [--trace-dir tmp/flashinfer-trace] \\
        [--peer-node-addr nvl72089-T16] \\
        [--extra-server-flag --disable-radix-cache] \\
        [--no-eval] \\
        [--no-push]

Notes:
  - --extra-server-flag defaults to --disable-cuda-graph.
    Ragged prefill: add --disable-radix-cache --disable-piecewise-cuda-graph
    Paged prefill: add --enable-deterministic-inference
  - Set --replace to overwrite existing workloads instead of appending.
  - --no-push performs a dry run (collects and sanitizes but does not upload).
  - --no-eval skips the flashinfer-bench eval and trace upload steps.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths (relative to repo root, resolved at startup)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_BENCH_SCRIPT = _REPO_ROOT / "examples" / "sglang_bench" / "bench_sharegpt.py"
_SANITIZE_SCRIPT = _REPO_ROOT / "scripts" / "sanitize_dumps.py"

HF_REPO_ID = "flashinfer-ai/flashinfer-trace"
HF_REPO_TYPE = "dataset"
HF_BATCH_SIZE = 500  # max operations per HF commit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[collect_stream {ts}] {msg}", flush=True)


def get_definition(trace_dir: Path, def_name: str) -> dict:
    matches = list((trace_dir / "definitions").glob(f"**/{def_name}.json"))
    if not matches:
        raise FileNotFoundError(f"Definition '{def_name}' not found under {trace_dir}/definitions/")
    with open(matches[0]) as f:
        return json.load(f)


def get_op_type(defn: dict) -> str:
    op = defn.get("op_type")
    if op:
        return op
    # Fallback: first component of name (e.g. "gqa_paged_decode_..." → "gqa_paged")
    name = defn["name"]
    for known in (
        "gqa_paged",
        "gqa_ragged",
        "mla_paged",
        "dsa_paged",
        "gdn",
        "moe",
        "rope",
        "rmsnorm",
        "gemm",
        "sampling",
        "mamba_ssu",
    ):
        if name.startswith(known):
            return known
    return name.split("_")[0]


def get_include_pattern(defn: dict) -> str:
    """Derive FLASHINFER_DUMP_INCLUDE glob from the fi_api tag."""
    for tag in defn.get("tags", []):
        if tag.startswith("fi_api:"):
            api = tag[len("fi_api:") :]
            parts = api.split(".")
            # Use the class name with a wildcard (matches .run, .plan, etc.)
            cls = next((p for p in parts if p[0].isupper()), None)
            if cls:
                return f"{cls}*"
            return parts[-1]
    raise ValueError(
        f"Definition '{defn['name']}' has no fi_api tag — cannot derive FLASHINFER_DUMP_INCLUDE. "
        "Pass --include-pattern explicitly."
    )


def snapshot_blobs(blob_dir: Path) -> set:
    """Return set of absolute safetensors paths currently in blob_dir."""
    if not blob_dir.exists():
        return set()
    return {str(p) for p in blob_dir.rglob("*.safetensors")}


# ---------------------------------------------------------------------------
# Step 1: run inference
# ---------------------------------------------------------------------------


def run_inference(
    def_name: str,
    model_key: str,
    model_path: str,
    batch_sizes: list,
    dump_dir: Path,
    include_pattern: str,
    num_batches: int,
    dump_max_count: int,
    dump_max_per_key: int,
    extra_server_flags: list,
    peer_node_addr: list,
    dist_init_port: int,
    conda_env: Optional[str],
    base_url: str,
) -> None:
    """Run bench_sharegpt.py once across all batch_sizes, collecting dumps into dump_dir."""
    cubins_dir = "/tmp/flashinfer_cubins"
    env = {
        **os.environ,
        # FlashInfer Level-10 logging
        "FLASHINFER_LOGLEVEL": "10",
        "FLASHINFER_DUMP_DIR": str(dump_dir),
        "FLASHINFER_DUMP_SAFETENSORS": "1",
        "FLASHINFER_DUMP_INCLUDE": include_pattern,
        "FLASHINFER_DUMP_EXCLUDE": "*.__init__",
        "FLASHINFER_DUMP_MAX_COUNT": str(dump_max_count),
        "FLASHINFER_DUMP_MAX_SIZE_GB": "16",
        "FLASHINFER_DUMP_DEDUP": "1",
        "FLASHINFER_DUMP_MAX_PER_KEY": str(dump_max_per_key),
        "FLASHINFER_USE_CUDA_NORM": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "FLASHINFER_CUBIN_DIR": cubins_dir,
        # SGLang tuning
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        "TRITON_CACHE_DIR": "/tmp/triton_cache_collect_stream",
    }

    cmd = [
        sys.executable,
        str(_BENCH_SCRIPT),
        "--model",
        model_key,
        "--model-path",
        model_path,
        "--batch-sizes",
        *[str(b) for b in batch_sizes],
        "--num-batches",
        str(num_batches),
        "--base-url",
        base_url,
    ] + extra_server_flags

    if peer_node_addr:
        for addr in peer_node_addr:
            cmd += ["--peer-node-addr", addr]
    if dist_init_port:
        cmd += ["--dist-init-port", str(dist_init_port)]
    if conda_env:
        cmd += ["--conda-env", conda_env]

    log(f"  CMD: {' '.join(cmd[:10])} ...")
    log(
        f"  FLASHINFER_DUMP_DIR={dump_dir}, DUMP_MAX_COUNT={dump_max_count}, "
        f"DEDUP=1 (max {dump_max_per_key} per shape)"
    )

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"bench_sharegpt.py exited {result.returncode} for batch_sizes={batch_sizes}"
        )

    n_dumps = sum(1 for _ in dump_dir.iterdir()) if dump_dir.exists() else 0
    log(f"  Inference done. Dump dir has {n_dumps} entries.")


# ---------------------------------------------------------------------------
# Step 2: sanitize
# ---------------------------------------------------------------------------


def run_sanitize(
    dump_dir: Path, def_name: str, trace_dir: Path, max_new_workloads: int, replace: bool = False
) -> None:
    cmd = [
        sys.executable,
        str(_SANITIZE_SCRIPT),
        "--dump-dir",
        str(dump_dir),
        "--definitions",
        def_name,
        "--flashinfer-trace-dir",
        str(trace_dir),
        "--max-new-workloads",
        str(max_new_workloads),
    ]
    if replace:
        cmd.append("--replace")

    log(f"  Sanitizing (max_new_workloads={max_new_workloads}, replace={replace}) ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"sanitize_dumps.py failed (exit {result.returncode})")


# ---------------------------------------------------------------------------
# Step 3: push to HF PR (append-only)
# ---------------------------------------------------------------------------


def push_to_pr(
    pr_num: int,
    def_name: str,
    op_type: str,
    trace_dir: Path,
    blob_dir: Path,
    old_blobs: set,
    batch_sizes: list,
) -> None:
    """Upload updated JSONL + new blobs to the HF PR.  Never deletes existing files."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()

    jsonl_path = trace_dir / "workloads" / op_type / f"{def_name}.jsonl"
    if not jsonl_path.exists():
        log(f"  WARNING: {jsonl_path} not found — nothing to push")
        return

    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
    log(f"  JSONL has {len(lines)} total workloads")

    new_blobs = snapshot_blobs(blob_dir) - old_blobs
    log(f"  New blob files: {len(new_blobs)}")

    # Always re-upload the full JSONL (it's small) so the HF copy stays current.
    ops = [
        CommitOperationAdd(
            path_in_repo=f"workloads/{op_type}/{def_name}.jsonl", path_or_fileobj=str(jsonl_path)
        )
    ]
    for blob_abs in sorted(new_blobs):
        blob_rel = Path(blob_abs).relative_to(trace_dir)
        ops.append(CommitOperationAdd(path_in_repo=str(blob_rel), path_or_fileobj=blob_abs))

    for i in range(0, len(ops), HF_BATCH_SIZE):
        batch = ops[i : i + HF_BATCH_SIZE]
        result = api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            operations=batch,
            commit_message=(
                f"Add {def_name} workloads (bs={','.join(str(b) for b in batch_sizes)}, "
                f"part {i // HF_BATCH_SIZE + 1})"
            ),
            revision=f"refs/pr/{pr_num}",
            num_threads=8,
        )
        log(f"  Pushed: {getattr(result, 'commit_url', result)}")


# ---------------------------------------------------------------------------
# Step 5: eval
# ---------------------------------------------------------------------------


def run_eval(def_name: str, trace_dir: Path) -> bool:
    """Run flashinfer-bench baseline evaluation. Returns True if all PASSED."""
    log("Running flashinfer-bench baseline eval ...")
    cmd = [
        "flashinfer-bench",
        "run",
        "--local",
        str(trace_dir),
        "--definitions",
        def_name,
        "--solutions",
        "baseline",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log(f"WARNING: eval exited {result.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Step 6: push trace
# ---------------------------------------------------------------------------


def push_trace(pr_num: int, def_name: str, op_type: str, trace_dir: Path) -> None:
    """Push the baseline trace JSONL to the HF PR."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()

    # flashinfer-bench writes traces to traces/{op_type}/{def_name}.jsonl
    trace_path = trace_dir / "traces" / op_type / f"{def_name}.jsonl"
    if not trace_path.exists():
        log(f"WARNING: trace not found at {trace_path}, skipping")
        return

    lines = [l for l in trace_path.read_text().splitlines() if l.strip()]
    log(f"Pushing trace ({len(lines)} entries) to PR #{pr_num} ...")

    result = api.create_commit(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"traces/{op_type}/{def_name}.jsonl", path_or_fileobj=str(trace_path)
            )
        ],
        commit_message=f"Add {def_name} baseline traces",
        revision=f"refs/pr/{pr_num}",
    )
    log(f"Trace pushed: {getattr(result, 'commit_url', result)}")
    log(f"PR2 complete: https://huggingface.co/datasets/{HF_REPO_ID}/discussions/{pr_num}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--def-name", required=True, help="Definition name (e.g. gqa_paged_decode_h5_kv1_d128_ps64)"
    )
    parser.add_argument(
        "--model-key",
        required=True,
        help="Model key for bench_sharegpt.py (e.g. llama-4-scout-ps64)",
    )
    parser.add_argument("--model-path", required=True, help="Path to model weights directory")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Batch sizes to collect (e.g. 64 128)",
    )
    parser.add_argument(
        "--pr-num", type=int, required=True, help="HuggingFace PR number to push workloads to"
    )
    parser.add_argument(
        "--trace-dir",
        default="tmp/flashinfer-trace",
        help="flashinfer-trace repo clone dir (default: tmp/flashinfer-trace)",
    )
    parser.add_argument(
        "--dump-base-dir",
        default="/tmp/fi_stream_dump",
        help="Base dir for the dump dir (default: /tmp/fi_stream_dump)",
    )
    parser.add_argument(
        "--dump-count",
        type=int,
        default=50,
        help=(
            "FLASHINFER_DUMP_MAX_COUNT (default: 50). With dedup mode each "
            "(batch_size, kv_length) shape signature is capped, so a small "
            "budget covers all batch sizes in one server session."
        ),
    )
    parser.add_argument(
        "--dump-max-per-key",
        type=int,
        default=2,
        help=(
            "FLASHINFER_DUMP_MAX_PER_KEY: dumps retained per unique input "
            "shape signature (default: 2)"
        ),
    )
    parser.add_argument(
        "--workloads-per-batch",
        type=int,
        default=4,
        help=(
            "Max new workloads to select per batch size; the sanitizer is "
            "called once over the combined dump dir with limit "
            "workloads_per_batch * len(batch_sizes) (default: 4)"
        ),
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Inference rounds across batch sizes (default: 1)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:20000",
        help="SGLang server base URL (default: http://127.0.0.1:20000)",
    )
    parser.add_argument(
        "--peer-node-addr",
        nargs="+",
        default=None,
        metavar="HOST",
        help="Peer node hostname(s) for multi-node TP",
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=20010,
        help="PyTorch distributed rendezvous port (default: 20010)",
    )
    parser.add_argument(
        "--conda-env", default=None, help="Conda environment name for peer-node SSH workers"
    )
    parser.add_argument(
        "--include-pattern",
        default=None,
        help=(
            "Override FLASHINFER_DUMP_INCLUDE pattern. " "Auto-derived from fi_api tag if omitted."
        ),
    )
    parser.add_argument(
        "--extra-server-flag",
        nargs="*",
        default=None,
        dest="extra_server_flags",
        metavar="FLAG",
        help=(
            "Extra flags forwarded verbatim to bench_sharegpt.py. "
            "Default: --disable-cuda-graph. "
            "Example: --extra-server-flag --disable-radix-cache --disable-piecewise-cuda-graph"
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace (not append) existing workloads",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip flashinfer-bench eval and trace upload (steps 5-6)",
    )
    parser.add_argument(
        "--no-push", action="store_true", help="Dry-run: collect and sanitize but do not push to HF"
    )
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).expanduser().resolve()
    dump_base_dir = Path(args.dump_base_dir)
    extra_server_flags = (
        args.extra_server_flags if args.extra_server_flags is not None else ["--disable-cuda-graph"]
    )

    # Load definition to derive op_type and include pattern
    defn = get_definition(trace_dir, args.def_name)
    op_type = get_op_type(defn)
    include_pattern = args.include_pattern or get_include_pattern(defn)
    blob_dir = trace_dir / "blob" / "workloads" / op_type / args.def_name

    total_workloads = args.workloads_per_batch * len(args.batch_sizes)
    log(f"Definition:    {args.def_name}")
    log(f"Op type:       {op_type}")
    log(f"Include:       {include_pattern}")
    log(f"Batch sizes:   {args.batch_sizes}")
    log(f"Dump limit:    {args.dump_count} (dedup max-per-key={args.dump_max_per_key})")
    log(f"Workloads:     {total_workloads} ({args.workloads_per_batch} × {len(args.batch_sizes)} batch sizes)")
    log(f"Num batches:   {args.num_batches}")
    log(f"HF PR:         #{args.pr_num}")
    log(f"Dry run:       {args.no_push}")

    dump_dir = dump_base_dir / args.def_name
    dump_dir.mkdir(parents=True, exist_ok=True)

    old_blobs = snapshot_blobs(blob_dir)

    # --- Step 1: inference (single server session over all batch sizes) ---
    log(f"\n{'=' * 60}")
    log("Step 1: inference")
    log(f"{'=' * 60}")
    run_inference(
        def_name=args.def_name,
        model_key=args.model_key,
        model_path=args.model_path,
        batch_sizes=args.batch_sizes,
        dump_dir=dump_dir,
        include_pattern=include_pattern,
        num_batches=args.num_batches,
        dump_max_count=args.dump_count,
        dump_max_per_key=args.dump_max_per_key,
        extra_server_flags=extra_server_flags,
        peer_node_addr=args.peer_node_addr or [],
        dist_init_port=args.dist_init_port,
        conda_env=args.conda_env,
        base_url=args.base_url,
    )

    # --- Step 2: sanitize ---
    log(f"\n{'=' * 60}")
    log("Step 2: sanitize")
    log(f"{'=' * 60}")
    run_sanitize(dump_dir, args.def_name, trace_dir, total_workloads, args.replace)

    jsonl_path = trace_dir / "workloads" / op_type / f"{args.def_name}.jsonl"
    n_total = 0
    if jsonl_path.exists():
        n_total = sum(1 for l in jsonl_path.read_text().splitlines() if l.strip())
    if n_total == 0:
        raise RuntimeError(
            f"0 workloads after sanitize — "
            "check FLASHINFER_DUMP_INCLUDE, plan() capture, and const-axis check"
        )
    log(f"  Workloads in JSONL: {n_total}")

    # --- Step 3: push ---
    log(f"\n{'=' * 60}")
    if not args.no_push:
        log("Step 3: push to HF PR")
        log(f"{'=' * 60}")
        push_to_pr(
            args.pr_num, args.def_name, op_type, trace_dir, blob_dir, old_blobs, args.batch_sizes
        )
    else:
        log("Step 3: skipped (--no-push)")
        log(f"{'=' * 60}")

    # --- Step 4: clear dump dir ---
    log(f"Step 4: clearing {dump_dir}")
    shutil.rmtree(dump_dir)

    if args.no_eval:
        log("Eval skipped (--no-eval). Done.")
        return

    # --- Step 5: eval ---
    log("Step 5: flashinfer-bench baseline eval")
    eval_ok = run_eval(args.def_name, trace_dir)
    if not eval_ok:
        log("ERROR: eval failed — not pushing trace. Fix failures and re-run with --no-eval=False.")
        sys.exit(1)

    # --- Step 6: push trace ---
    if not args.no_push:
        log("Step 6: push trace to HF PR")
        push_trace(args.pr_num, args.def_name, op_type, trace_dir)
    else:
        log("Step 6: skipped (--no-push)")

    log("\nDone.")


if __name__ == "__main__":
    main()
