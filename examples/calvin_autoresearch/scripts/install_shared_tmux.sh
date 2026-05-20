#!/usr/bin/env bash
set -euo pipefail

# Install a shared tmux into /public/seven without requiring root.
# Run this from a session where /public/seven is writable and the network is
# available.  The installed tmux is isolated from the StarVLA Python env.

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/26220172}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
TMUX_PREFIX="${TMUX_PREFIX:-${SHARED_ROOT}/shared/tools/tmux}"
ENV_FILE="${ENV_FILE:-${SHARED_ROOT}/shared/env/tmux.shared.sh}"

mkdir -p "$(dirname "${TMUX_PREFIX}")" "$(dirname "${ENV_FILE}")"

if [[ -x "${TMUX_PREFIX}/bin/tmux" ]]; then
  echo "[tmux] already installed: ${TMUX_PREFIX}/bin/tmux"
  "${TMUX_PREFIX}/bin/tmux" -V
else
  installer=""
  if [[ -n "${MICROMAMBA:-}" && -x "${MICROMAMBA}" ]]; then
    installer="${MICROMAMBA}"
  elif command -v micromamba >/dev/null 2>&1; then
    installer="$(command -v micromamba)"
  elif [[ -x "${PROJECT_ROOT}/tools/micromamba/bin/micromamba" ]]; then
    installer="${PROJECT_ROOT}/tools/micromamba/bin/micromamba"
  elif command -v mamba >/dev/null 2>&1; then
    installer="$(command -v mamba)"
  elif command -v conda >/dev/null 2>&1; then
    installer="$(command -v conda)"
  fi

  if [[ -z "${installer}" ]]; then
    echo "[tmux] missing micromamba/mamba/conda; cannot install without root." >&2
    exit 2
  fi

  echo "[tmux] installer: ${installer}"
  echo "[tmux] prefix: ${TMUX_PREFIX}"

  case "$(basename "${installer}")" in
    micromamba)
      export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${SHARED_ROOT}/shared/tools/micromamba-root}"
      "${installer}" create -y -p "${TMUX_PREFIX}" -c conda-forge tmux
      ;;
    mamba|conda)
      "${installer}" create -y -p "${TMUX_PREFIX}" -c conda-forge tmux
      ;;
    *)
      echo "[tmux] unsupported installer: ${installer}" >&2
      exit 2
      ;;
  esac
fi

cat > "${ENV_FILE}" <<EOF
#!/usr/bin/env bash
export SEVEN_STARVLA_TMUX_PREFIX="${TMUX_PREFIX}"
export PATH="\${SEVEN_STARVLA_TMUX_PREFIX}/bin:\${PATH}"
EOF
chmod +x "${ENV_FILE}"

echo "[tmux] env file: ${ENV_FILE}"
echo "[tmux] activate with:"
echo "  source ${ENV_FILE}"
echo "[tmux] version:"
"${TMUX_PREFIX}/bin/tmux" -V
