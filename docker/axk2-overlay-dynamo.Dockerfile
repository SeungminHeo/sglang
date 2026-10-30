# Dynamo runtime variant of the overlay image (axk2-overlay.Dockerfile), on ai-dynamo 1.6.x.
#
# The lmsysorg nightly-dev base ships ai-dynamo 1.7.0.dev; this swaps it for the
# 1.6 release line. `ai-dynamo[sglang]` pins `sglang==0.5.21`, which would
# replace the overlaid main-based sglang, so install `ai-dynamo` without the
# extra and keep every other base package pinned (the extra's runtime deps —
# accelerate, blake3, cupy, nixl, uvloop — are already in the base). Only the
# Dynamo pair and what ai-dynamo 1.6.1 itself needs to move (certifi>=2026.10.19,
# aisimulate's googleapis-common-protos / pydantic bounds) are left unpinned.
#
# NOTE: Dynamo 1.6.1 is validated upstream against sglang 0.5.21; the branch
# sits on upstream main 586355a8, so smoke-test `python -m dynamo.sglang`
# (aggregated + disagg prefill/decode) after any Dynamo or branch bump.
#
#   docker build --platform linux/amd64 --build-arg BASE_IMAGE=<overlay image> \
#     -f docker/axk2-overlay-dynamo.Dockerfile -t <tag> docker
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DYNAMO_VERSION=1.6.1
# uv segfaults under qemu user emulation (local cross-arch builds).
ARG INSTALLER="python3 -m pip install --no-cache-dir"

RUN python3 -m pip list --format=freeze \
        | grep -E '^[A-Za-z0-9_.-]+==' \
        | grep -viE '^(ai-dynamo|ai-dynamo-runtime|aisimulate|pydantic(-core|_core)?|certifi|googleapis-common-protos)==' \
        > /tmp/base-constraints.txt \
    && ${INSTALLER} -c /tmp/base-constraints.txt "ai-dynamo==${DYNAMO_VERSION}" \
    && rm /tmp/base-constraints.txt \
    && python3 -c "\
import importlib.metadata as md, importlib.util; \
assert md.version('ai-dynamo') == '${DYNAMO_VERSION}', md.version('ai-dynamo'); \
assert md.version('ai-dynamo-runtime') == md.version('ai-dynamo'); \
import dynamo.runtime; \
assert all(importlib.util.find_spec(m) for m in ('dynamo.sglang', 'dynamo.frontend')); \
print('ai-dynamo', md.version('ai-dynamo'), 'sglang', md.version('sglang'), 'nixl', md.version('nixl-cu13'))"

# Dynamo 1.6.1's SGLang chat processor ignores Solar Open 2's reasoning_effort
# (effort low/none answers land in reasoning_content); see the script.
COPY patches/dynamo-sglang-frontend.py /tmp/dynamo-sglang-frontend.py
RUN python3 /tmp/dynamo-sglang-frontend.py && rm /tmp/dynamo-sglang-frontend.py

# The Dynamo operator passes full commands
# (`python3 -m dynamo.sglang ...`, `python3 -m dynamo.frontend ...`).
ENTRYPOINT []
CMD ["/bin/bash"]

LABEL ai.skt.dynamo.version="${DYNAMO_VERSION}"
