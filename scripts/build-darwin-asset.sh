#!/usr/bin/env bash
# Builds the self-contained darwin-arm64 tarball attached to each GitHub release.
#
# Tarball contract:
#   - asset name:  claudex-gateway-<version>-darwin-arm64.tar.gz
#   - tar root:    bin/claudex-gateway executable entrypoint
#
# The gateway is assembled from a wheel, bundled CPython, and darwin-arm64
# dependency wheels on macOS or a cross-build host such as Linux arm64:
#   bin/claudex-gateway   POSIX shell shim that execs the bundled runtime
#   bin/claudex           launches Claude Code through the local gateway
#   python/               python-build-standalone darwin-arm64 (checksum-pinned)
#   python/lib/.../site-packages
#                         project wheel + uv.lock-pinned deps, cross-installed
#
# Every native library, bundled Python, and Playwright Node must support
# Mach-O arm64. Darwin arm64 uses lipo and runs the bundled runtime, MCP, and
# Playwright driver. Other hosts use a portable Mach-O check and host CPython
# of the same series for a CLI usage smoke; bundled-runtime execution is skipped.
# Cross-builds require uv to locate or download that host CPython. The resulting
# asset still runs only on macOS arm64, regardless of its build host.
#
# Output: build/claudex-gateway-<version>-darwin-arm64.tar.gz
set -euo pipefail

# Pinned bundled runtime: python-build-standalone "install_only" build.
# Update all four values together; the sha256 is listed in the release's
# SHA256SUMS file.
PBS_TAG="20260728"
PBS_PYTHON="cpython-3.12.13"
PBS_SERIES="3.12"
PBS_SHA256="12d6700f7e8f222639f0ee5bbd173082c3041aeb65af8f9828e4216bc8047de6"
# Swap to the Nexus raw proxy if the build host cannot reach github.com.
PBS_BASE_URL="${PBS_BASE_URL:-https://github.com/astral-sh/python-build-standalone/releases/download}"

TOOL="claudex-gateway"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail() { echo "error: $1" >&2; exit 1; }

command -v uv >/dev/null 2>&1 \
  || fail "uv is required to build the asset (https://docs.astral.sh/uv/)"
HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"
IS_NATIVE_HOST=false
if [ "${HOST_OS}" = "Darwin" ] && [ "${HOST_ARCH}" = "arm64" ]; then
  IS_NATIVE_HOST=true
  command -v lipo >/dev/null 2>&1 \
    || fail "lipo is required to verify arm64 native dependencies (install Xcode command line tools)"
else
  HOST_PYTHON="$(env -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH \
    uv run --no-project --python "cpython@${PBS_SERIES}" \
    python -S -c 'import sys; print(sys.executable)')" \
    || fail "could not obtain host CPython ${PBS_SERIES} through uv for cross-build verification"
  [ -x "${HOST_PYTHON}" ] || fail "host CPython is not executable: ${HOST_PYTHON}"
fi

VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -1)"
[ -n "${VERSION}" ] || fail "could not read version from pyproject.toml"
ASSET="${TOOL}-${VERSION}-darwin-arm64.tar.gz"

STAGE="build/stage"
SITE_PACKAGES="${STAGE}/python/lib/python${PBS_SERIES}/site-packages"
rm -rf build dist
mkdir -p build/downloads "${STAGE}"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

echo "==> Fetching bundled CPython (${PBS_PYTHON}, darwin-arm64)"
PBS_TARBALL="${PBS_PYTHON}+${PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz"
curl -fsSL -o "build/downloads/${PBS_TARBALL}" "${PBS_BASE_URL}/${PBS_TAG}/${PBS_TARBALL}"
GOT_SHA256="$(sha256_of "build/downloads/${PBS_TARBALL}")"
[ "${GOT_SHA256}" = "${PBS_SHA256}" ] \
  || fail "checksum mismatch for ${PBS_TARBALL} — corrupted download or tampered mirror (${PBS_BASE_URL}): got ${GOT_SHA256}"
# The install_only tarball extracts to a python/ root — exactly the layout
# the shim expects next to bin/.
tar -xzf "build/downloads/${PBS_TARBALL}" -C "${STAGE}"
[ -x "${STAGE}/python/bin/python3" ] || fail "unexpected runtime layout: missing python/bin/python3"

echo "==> Cross-installing project wheel + uv.lock-pinned deps into the bundled site-packages"
uv build --wheel
uv export --frozen --no-dev --no-emit-project --no-hashes -o build/requirements.txt
uv pip install \
  --target "${SITE_PACKAGES}" \
  --python-version "${PBS_SERIES}" \
  --python-platform aarch64-apple-darwin \
  --only-binary :all: \
  -r build/requirements.txt \
  "dist/claudex_gateway-${VERSION}-py3-none-any.whl"
# Console scripts are unused — the shim runs `python -m claudex`.
rm -rf "${SITE_PACKAGES}/bin"

echo "==> Verifying native dependency architecture"
PLAYWRIGHT_NODE="${SITE_PACKAGES}/playwright/driver/node"
[ -x "${PLAYWRIGHT_NODE}" ] || fail "missing or non-executable Playwright driver: ${PLAYWRIGHT_NODE}"
native_files() {
  printf '%s\0' "${STAGE}/python/bin/python3" "${PLAYWRIGHT_NODE}"
  find "${STAGE}/python" \( -name '*.so' -o -name '*.dylib' \) -print0
}
if [ "${IS_NATIVE_HOST}" = true ]; then
  native_files |
    while IFS= read -r -d '' native_file; do
      lipo "${native_file}" -verify_arch arm64 >/dev/null 2>&1 \
        || fail "native dependency lacks arm64 support: ${native_file}"
    done
else
  native_files | "${HOST_PYTHON}" -S "${ROOT}/scripts/verify_darwin_architecture.py"
fi

echo "==> Writing launcher shims"
mkdir -p "${STAGE}/bin"
cat > "${STAGE}/bin/${TOOL}" <<'EOF'
#!/bin/sh
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/python/bin/python3" -m claudex "$@"
EOF
chmod +x "${STAGE}/bin/${TOOL}"
# The tarball also ships a ready-made `claudex` command that starts the
# gateway (idempotent background start) and launches Claude Code through it;
# `claudex settings` (exactly that
# one argument) opens the dashboard instead — everything else passes through
# to claude untouched. Success paths stay quiet (stdout dropped); startup
# failures surface on stderr and abort.
cat > "${STAGE}/bin/claudex" <<'EOF'
#!/bin/sh
# Launches Claude Code through the local claudex-gateway, starting it if needed.
# `claudex settings` opens the gateway dashboard in the browser.
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/claudex-gateway" >/dev/null || exit 1
BASE_URL="http://${CLAUDEX_HOST:-127.0.0.1}:${CLAUDEX_PORT:-8787}"
if [ "$#" -eq 1 ] && [ "$1" = "settings" ]; then
  exec open "$BASE_URL/"
fi
command -v claude >/dev/null 2>&1 || { echo "claudex: claude (Claude Code) is not on PATH" >&2; exit 127; }
exec env ANTHROPIC_BASE_URL="$BASE_URL" claude "$@"
EOF
chmod +x "${STAGE}/bin/claudex"

echo "==> Verifying assembled layout"
[ -x "${STAGE}/bin/${TOOL}" ] || fail "${STAGE}/bin/${TOOL} is missing or not executable"
[ -x "${STAGE}/bin/claudex" ] || fail "${STAGE}/bin/claudex is missing or not executable"
[ -f "${SITE_PACKAGES}/claudex/__main__.py" ] || fail "claudex package missing from site-packages"

# Smoke tests must not use developer configuration or add host-specific bytecode
# to the archive. Never start the gateway server or a browser during the build.
SMOKE_HOME="${PWD}/build/smoke-home"
mkdir -p "${SMOKE_HOME}"
run_smoke() (
  for variable in "${!CLAUDEX_@}"; do unset "${variable}"; done
  unset PYTHONHOME PLAYWRIGHT_NODEJS_PATH
  export HOME="${SMOKE_HOME}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  cd "${SMOKE_HOME}" || exit 1
  "$@"
)
SMOKE_RC=0
if [ "${IS_NATIVE_HOST}" = true ]; then
  echo "==> Smoke-testing the bundled runtime"
  SMOKE_OUT="$(PYTHONPATH= run_smoke "${ROOT}/${STAGE}/bin/${TOOL}" \
    definitely-not-a-subcommand 2>&1)" || SMOKE_RC=$?
else
  echo "==> Skipping bundled Darwin runtime verification on ${HOST_OS} ${HOST_ARCH} (Python/launcher, MCP, Playwright driver)"
  echo "==> Smoke-testing platform-independent CLI with host CPython ${PBS_SERIES}"
  SMOKE_OUT="$(PYTHONPATH="${ROOT}/${SITE_PACKAGES}" run_smoke "${HOST_PYTHON}" \
    -S -m claudex definitely-not-a-subcommand 2>&1)" || SMOKE_RC=$?
fi
[ "${SMOKE_RC}" = "2" ] || fail "smoke test exited with ${SMOKE_RC}, expected usage error 2: ${SMOKE_OUT}"
case "${SMOKE_OUT}" in
  *"usage: claudex-gateway"*) ;;
  *) fail "smoke test did not print the usage line: ${SMOKE_OUT}" ;;
esac
if [ "${IS_NATIVE_HOST}" = true ]; then
  PYTHONPATH= run_smoke "${ROOT}/${STAGE}/python/bin/python3" - <<'PY'
import asyncio

from claudex.mcp_tools import build_gptpro_server
from playwright.async_api import async_playwright

build_gptpro_server(None)

async def verify_driver():
    playwright = await async_playwright().start()
    try:
        assert playwright.chromium.executable_path
    finally:
        await playwright.stop()

asyncio.run(verify_driver())
print("bundled MCP import and Playwright driver startup passed")
PY
fi

echo "==> Packing ${ASSET}"
# COPYFILE_DISABLE keeps macOS builds from adding ._* AppleDouble entries.
(cd "${STAGE}" && COPYFILE_DISABLE=1 tar -czf "../${ASSET}" .)
# Final structure check against the tarball contract. The listing is grepped
# from a file: piping it into `grep -q` dies of SIGPIPE under pipefail when
# grep exits on an early match, misreporting present entries as missing.
LISTING_FILE="build/asset-listing.txt"
tar -tzf "build/${ASSET}" > "${LISTING_FILE}"
for entry in "./bin/${TOOL}" "./bin/claudex" "./python/bin/python3" "./python/lib/python${PBS_SERIES}/site-packages/claudex/__main__.py"; do
  grep -qx "${entry}" "${LISTING_FILE}" || fail "build/${ASSET} is missing ${entry}"
done
rm -f "${LISTING_FILE}"

echo "==> Done"
ls -lh "build/${ASSET}"
echo "sha256: $(sha256_of "build/${ASSET}")"
