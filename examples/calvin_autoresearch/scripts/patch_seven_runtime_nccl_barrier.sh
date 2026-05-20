#!/usr/bin/env bash
set -euo pipefail

# Patch the exported public/seven runtime in place.
# Fixes:
#   1. robust MEMBER default under `set -u`
#   2. NCCL dist.barrier() before PyTorch knows the local CUDA device

RUNTIME_ROOT="${RUNTIME_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/runtime}"
STARVLA_ROOT="${RUNTIME_ROOT}/code/starVLA"

log() {
  printf '[patch-seven-runtime] %s\n' "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'Missing file: %s\n' "${path}" >&2
    exit 2
  fi
}

patch_launchers() {
  local file
  for file in \
    "${RUNTIME_ROOT}/scripts/train_abc_headonly_h200.sh" \
    "${RUNTIME_ROOT}/scripts/train_abc_headonly_from_wmh_60k_h200.sh"
  do
    require_file "${file}"
    if grep -q 'MEMBER_CANDIDATE' "${file}"; then
      log "launcher already patched: ${file}"
      continue
    fi
    python - "${file}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = 'MEMBER="${MEMBER:-${SUDO_USER:-${USER:-shared}}}"'
new = '''MEMBER_CANDIDATE="${MEMBER-}"
if [[ -z "${MEMBER_CANDIDATE}" ]]; then
  MEMBER_CANDIDATE="${SUDO_USER-}"
fi
if [[ -z "${MEMBER_CANDIDATE}" ]]; then
  MEMBER_CANDIDATE="${USER-}"
fi
MEMBER="${MEMBER_CANDIDATE:-shared}"'''
if old not in text:
    raise SystemExit(f"pattern not found in {path}: {old}")
path.write_text(text.replace(old, new, 1))
PY
    chmod a+rx "${file}"
    log "patched launcher: ${file}"
  done
}

patch_train_starvla() {
  local file="${STARVLA_ROOT}/starVLA/training/train_starvla.py"
  require_file "${file}"
  python - "${file}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

if "_set_local_cuda_device()" not in text:
    anchor = "# Local Modules\n"
    block = '''# Bind each worker to its local CUDA device before any NCCL barrier.
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None and torch.cuda.is_available():
    _local_rank_int = int(_local_rank)
    if 0 <= _local_rank_int < torch.cuda.device_count():
        torch.cuda.set_device(_local_rank_int)
del _local_rank

'''
    if anchor not in text:
        raise SystemExit(f"anchor not found in {path}: {anchor!r}")
    text = text.replace(anchor, block + anchor, 1)

if "def _dist_barrier()" not in text:
    anchor = 'logger = get_logger(__name__)\n'
    block = '''

def _dist_barrier():
    if not dist.is_available() or not dist.is_initialized():
        return
    barrier = dist.barrier
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        barrier(device_ids=[torch.cuda.current_device()])
    else:
        barrier()
'''
    if anchor not in text:
        raise SystemExit(f"anchor not found in {path}: {anchor!r}")
    text = text.replace(anchor, anchor + block, 1)

text = text.replace("dist.barrier()", "_dist_barrier()")
path.write_text(text)
PY
  log "patched train_starvla barriers: ${file}"
}

patch_lerobot_dataset() {
  local file="${STARVLA_ROOT}/starVLA/dataloader/gr00t_lerobot/datasets.py"
  require_file "${file}"
  python - "${file}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

if "def _dist_barrier()" not in text:
    anchor = "EPSILON = 5e-4\n"
    block = '''

def _dist_barrier():
    if not dist.is_available() or not dist.is_initialized():
        return
    barrier = dist.barrier
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            local_rank_int = int(local_rank)
            if 0 <= local_rank_int < torch.cuda.device_count():
                torch.cuda.set_device(local_rank_int)
        barrier(device_ids=[torch.cuda.current_device()])
    else:
        barrier()
'''
    if anchor not in text:
        raise SystemExit(f"anchor not found in {path}: {anchor!r}")
    text = text.replace(anchor, anchor + block, 1)

text = text.replace("dist.barrier()", "_dist_barrier()")
path.write_text(text)
PY
  log "patched dataset barriers: ${file}"
}

main() {
  require_file "${STARVLA_ROOT}/starVLA/training/train_starvla.py"
  require_file "${STARVLA_ROOT}/starVLA/dataloader/gr00t_lerobot/datasets.py"
  patch_launchers
  patch_train_starvla
  patch_lerobot_dataset
  chmod -R a+rX "${RUNTIME_ROOT}"
  log "done"
}

main "$@"
