#!/usr/bin/env bash
# Builds the self-contained darwin-arm64 tarball attached to each GitHub release.
#
# Tarball contract:
#   - asset name:  claudex-gateway-<version>-darwin-arm64.tar.gz
#   - tar root:    bin/claudex-gateway executable entrypoint
#
# The gateway is assembled from a wheel, bundled CPython, and darwin-arm64
# dependency wheels on an arm64 Mac:
#   bin/claudex-gateway   POSIX shell shim that execs the bundled runtime
#   bin/claudex           launches Claude Code through the local gateway
#   python/               python-build-standalone darwin-arm64 (checksum-pinned)
#   python/lib/.../site-packages
#                         project wheel + uv.lock-pinned deps, cross-installed
#
# Native dependencies are accepted only when they support arm64; the bundled
# runtime smoke test verifies they load without relying on the build host.
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
[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] \
  || fail "building the gptpro release asset requires a Darwin arm64 host"
command -v lipo >/dev/null 2>&1 \
  || fail "lipo is required to verify arm64 native dependencies (install Xcode command line tools)"

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
find "${SITE_PACKAGES}" \( -name '*.so' -o -name '*.dylib' \) -print0 |
  while IFS= read -r -d '' native_file; do
    lipo "${native_file}" -verify_arch arm64 >/dev/null 2>&1 \
      || fail "native dependency lacks arm64 support: ${native_file}"
  done

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
# The python binary must be Mach-O arm64. `file` may be absent on the CI
# container — check the Mach-O 64-bit magic (cf fa ed fe) and arm64 cputype
# (0c 00 00 01) directly.
HEADER="$(head -c 8 "${STAGE}/python/bin/python3" | od -An -tx1 | tr -d ' \n')"
[ "${HEADER}" = "cffaedfe0c000001" ] \
  || fail "python/bin/python3 is not Mach-O arm64 (header: ${HEADER})"

echo "==> Smoke-testing the bundled runtime"
# Use a fresh home and disable host Python packages; never start the gateway
# server or a browser during the build.
SMOKE_HOME="${PWD}/build/smoke-home"
mkdir -p "${SMOKE_HOME}"
SMOKE_RC=0
SMOKE_OUT="$(HOME="${SMOKE_HOME}" PYTHONPATH= PYTHONNOUSERSITE=1 \
  "${STAGE}/bin/${TOOL}" definitely-not-a-subcommand 2>&1)" || SMOKE_RC=$?
[ "${SMOKE_RC}" = "2" ] || fail "smoke test exited with ${SMOKE_RC}, expected usage error 2: ${SMOKE_OUT}"
case "${SMOKE_OUT}" in
  *"usage: claudex-gateway"*) ;;
  *) fail "smoke test did not print the usage line: ${SMOKE_OUT}" ;;
esac
HOME="${SMOKE_HOME}" PYTHONPATH= PYTHONNOUSERSITE=1 \
  "${STAGE}/python/bin/python3" - <<'PY'
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
