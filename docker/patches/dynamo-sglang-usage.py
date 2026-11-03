"""Report usage.completion_tokens_details.reasoning_tokens on the Dynamo 1.6.1 SGLang chat processor.

The SGLang worker sends no completion_tokens_details and SGLang's reasoning parser
has no token counter, so every response through `dynamo.frontend` with the SGLang
chat processor reports no reasoning_tokens. Count them from the generated token ids
in the processor, starting from the request parser's initial think state (Solar
Open 2 prefills the think start in the prompt), and add the field to the usage
chunk. A positive count from the backend wins.

Applied at image build time; fails if the anchors moved.
"""
import pathlib
import sys

import dynamo.frontend.sglang_processor as mod

path = pathlib.Path(mod.__file__)
src = path.read_text()

CLASS_ANCHOR = """logger = logging.getLogger(__name__)


"""
CLASS_PATCH = CLASS_ANCHOR + '''class _ReasoningTokenCounter:
    """Report reasoning tokens on the SGLang chat-processor path (AXIOM image patch).

    The SGLang worker sends no completion_tokens_details and SGLang's reasoning
    parser has no token counter, so count the original generated ids here. Text
    is never re-encoded (detokenisation is not injective). The end marker counts,
    matching SGLang's own Req.update_reasoning_tokens. The start state comes from
    the request's parser before it sees any output.
    """

    __slots__ = ("total", "_start_id", "_end_id", "_in_reasoning")

    def __init__(self, tokenizer: Any, reasoning_parser: Any) -> None:
        detector = getattr(reasoning_parser, "detector", None)
        self.total = 0
        self._start_id = self._single_token_id(
            tokenizer, getattr(detector, "think_start_token", None)
        )
        self._end_id = self._single_token_id(
            tokenizer, getattr(detector, "think_end_token", None)
        )
        self._in_reasoning = bool(getattr(detector, "_in_reasoning", False))

    @staticmethod
    def _single_token_id(tokenizer: Any, token: str | None) -> int | None:
        if not token:
            return None
        ids = tokenizer.encode(token, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def add(self, token_ids: Any) -> None:
        if self._end_id is None:
            return
        for token_id in token_ids:
            if self._in_reasoning:
                self.total += 1
                if token_id == self._end_id:
                    self._in_reasoning = False
            elif token_id == self._start_id:
                self.total += 1
                self._in_reasoning = True

    def annotate(self, usage: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of usage with reasoning_tokens; a positive backend count wins."""
        annotated = dict(usage)
        details = dict(annotated.get("completion_tokens_details") or {})
        backend = details.get("reasoning_tokens")
        if not isinstance(backend, int) or backend <= 0:
            details["reasoning_tokens"] = min(
                self.total, annotated.get("completion_tokens", self.total)
            )
        annotated["completion_tokens_details"] = details
        return annotated


'''

COUNTER_ANCHOR = """            latest_usage: dict[str, Any] = {}
            first_chunk = True
"""
COUNTER_PATCH = """            latest_usage: dict[str, Any] = {}
            reasoning_counter = _ReasoningTokenCounter(
                self.tokenizer, post.reasoning_parser
            )
            first_chunk = True
"""

USAGE_ANCHOR = """                        dynamo_out["usage"] = usage
"""
USAGE_PATCH = """                        dynamo_out["usage"] = reasoning_counter.annotate(usage)
"""

ADD_ANCHOR = """                cumulative_output_tokens += chunk_tokens
                raw_finish_reason = engine_response.get("finish_reason")
"""
ADD_PATCH = """                cumulative_output_tokens += chunk_tokens
                reasoning_counter.add(new_ids)
                raw_finish_reason = engine_response.get("finish_reason")
"""

if "_ReasoningTokenCounter" in src:
    sys.exit(f"{path} is already patched")
for anchor, patch in (
    (CLASS_ANCHOR, CLASS_PATCH),
    (COUNTER_ANCHOR, COUNTER_PATCH),
    (USAGE_ANCHOR, USAGE_PATCH),
    (ADD_ANCHOR, ADD_PATCH),
):
    assert src.count(anchor) == 1, f"anchor not unique in {path}: {anchor[:60]!r}"
    src = src.replace(anchor, patch)
path.write_text(src)
print("patched", path)
