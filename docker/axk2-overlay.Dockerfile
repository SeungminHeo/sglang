# Test image for the A.X-K2 port: official v0.5.16 release image (precompiled
# sgl-kernel / flashinfer, amd64) + this branch's pure-Python changes.
#
# The overlay copies source files over the installed sglang package instead of
# pip-installing it: the port adds no kernels, this branch is exactly v0.5.16
# plus the AXK2 commits, and a pip wheel build would need the Rust toolchain
# (absent in the image) for sglang's Rust extension — which the copy keeps
# untouched from the base image.
#
# Build from a checkout of the support-axk2-v0.5.16 branch, on/for amd64:
#   docker build --platform linux/amd64 \
#     -f docker/axk2-overlay.Dockerfile -t sglang-axk2:v0.5.16 .
#
# BASE_TAG=v0.5.16-cu130 for CUDA 13 hosts (driver >= 580).
ARG BASE_TAG=v0.5.16-cu129
FROM lmsysorg/sglang:${BASE_TAG}

COPY python/sglang /tmp/sglang-overlay
RUN target=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent)") \
    && echo "overlaying onto $target" \
    && cp -a /tmp/sglang-overlay/. "$target"/ \
    && rm -rf /tmp/sglang-overlay

# Fail the build early if the model did not register.
RUN python3 -c "from sglang.srt.models.axk2 import AXK2ForCausalLM; \
from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY; \
assert 'axk2' in _CONFIG_REGISTRY; print('AXK2 registered OK')"
