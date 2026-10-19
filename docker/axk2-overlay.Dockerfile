# Test image for the A.X-K2 port: an official lmsysorg/sglang image
# (precompiled sgl-kernel / flashinfer / Rust extensions, amd64) + this
# branch's pure-Python changes.
#
# The overlay copies source files over the installed sglang package instead of
# pip-installing it: the port adds no kernels, and a pip wheel build would need
# the Rust toolchain (absent in the image) for sglang's Rust extension — which
# the copy keeps untouched from the base image.
#
# BASE_TAG must be an image whose pinned dependencies (python/pyproject.toml)
# match this branch's base commit:
#   - main-based branch: the nightly-dev-cu13-<date>-<sha> image built from the
#     nearest upstream main commit with an identical pyproject.toml.
#   - release branch:    v<release>-cu130.
# Use a -cu130 / cu13 tag on CUDA 13 hosts (driver >= 580).
#
#   docker build --platform linux/amd64 --build-arg BASE_TAG=<tag> \
#     -f docker/axk2-overlay.Dockerfile -t sglang-axk2:<tag> .
ARG BASE_TAG
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
