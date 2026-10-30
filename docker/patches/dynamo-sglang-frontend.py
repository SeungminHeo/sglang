"""Teach the Dynamo 1.6.1 SGLang chat processor about Solar Open 2's reasoning effort.

The frontend builds ReasoningParser without the request, so SolarOpen2Detector never
sees reasoning_effort and treats every response as thinking: with effort low/none
(the template prefills a closed think block) the whole answer, json_schema output
included, lands in reasoning_content and content is empty. It also resolves
force_reasoning from the static template default, which decides whether guided
decoding waits for the think block to close.

This mirrors the native SGLang server: the effort the frontend renders into the
prompt (request field first, then chat_template_kwargs, as in
_render_prompt_token_ids) decides thinking, and reaches the detector.

Two post-processor fixes ride along:
- Output cut short (finish_reason length/abort/error) never becomes executable
  Solar Open 2 tool calls. Calls are emitted at finish from a re-parse, so a
  truncated `required` JSON array or a second native call cut mid-way would
  otherwise still surface the calls completed before the cut.
- A guided (`required`/named) response that thinks first and arrives in one
  engine chunk kept no reasoning_content: the JSON-array shortcut handed the
  whole buffer, reasoning included, to the tool parser.

Applied at image build time; fails if the anchors moved.
"""
import pathlib
import sys

import dynamo.frontend.sglang_prepost as mod

path = pathlib.Path(mod.__file__)
src = path.read_text()

EFFORT_HELPER = '''

def _solar_open2_effort(request: dict[str, Any]) -> Any:
    """The reasoning_effort the prompt is rendered with (request field wins)."""
    effort = request.get("reasoning_effort")
    if effort is None:
        kwargs = (
            request.get("chat_template_kwargs")
            or request.get("chat_template_args")
            or {}
        )
        effort = kwargs.get("reasoning_effort")
    return effort


def resolve_request_force_reasoning('''

RESOLVE_ANCHOR = '''    if reasoning_parser_name == "minimax-m3":
        return kwargs.get("thinking_mode") != "disabled"
'''
RESOLVE_PATCH = RESOLVE_ANCHOR + '''
    # Solar Open 2 thinks unless reasoning_effort is low/none (template default: high).
    if reasoning_parser_name == "solar_open2":
        return SolarOpen2Detector.thinks(_solar_open2_effort(request))
'''

PARSER_ANCHOR = '''            force_reasoning=force_reasoning,
            **_parser_tokenizer_kwargs(ReasoningParser, tokenizer),
        )
'''
PARSER_PATCH = '''            force_reasoning=force_reasoning,
            **_parser_tokenizer_kwargs(ReasoningParser, tokenizer),
            # SolarOpen2Detector picks its mode from the request's effort.
            **(
                {
                    "request": SimpleNamespace(
                        reasoning_effort=_solar_open2_effort(request),
                        chat_template_kwargs={},
                    )
                }
                if reasoning_parser_name == "solar_open2"
                else {}
            ),
        )
'''

IMPORT_ANCHOR = "from sglang.srt.parser.reasoning_parser import GptOssDetector, ReasoningParser\n"
IMPORT_PATCH = (
    "from types import SimpleNamespace\n\n"
    "from sglang.srt.parser.reasoning_parser import (\n"
    "    GptOssDetector,\n"
    "    ReasoningParser,\n"
    "    SolarOpen2Detector,\n"
    ")\n"
)

SHORTCUT_ANCHOR = """        self._pending_guided_reasoning_parts = None
        if finish_reason and _try_parse_json_array(buffered) is not None:
            return None, buffered
"""
SHORTCUT_PATCH = """        self._pending_guided_reasoning_parts = None
        # A closed think block ahead of the array goes through the reasoning parser.
        think_end = getattr(detector, "think_end_token", "")
        if (
            finish_reason
            and not (think_end and think_end in buffered)
            and _try_parse_json_array(buffered) is not None
        ):
            return None, buffered
"""

EMIT_ANCHOR = """        if finish_reason and self._tool_call_names:
            tool_calls_out: list[dict[str, Any]] = []
"""
EMIT_PATCH = """        if (
            finish_reason not in (None, "stop")
            and self._tool_call_names
            and self._tool_call_parser_name == "solar_open2"
        ):
            # Output cut short is not an executable call, even where the
            # calls before the cut parsed.
            logger.warning(
                "Dropping Solar Open 2 tool calls from output finished by %r: %s",
                finish_reason,
                sorted(self._tool_call_names.values()),
            )
            self._tool_call_names.clear()
            self._tool_call_ids.clear()
            self._tool_call_args.clear()

""" + EMIT_ANCHOR

if "_solar_open2_effort" in src:
    sys.exit(f"{path} is already patched")
for anchor, patch in (
    (IMPORT_ANCHOR, IMPORT_PATCH),
    ("\n\ndef resolve_request_force_reasoning(", EFFORT_HELPER),
    (RESOLVE_ANCHOR, RESOLVE_PATCH),
    (PARSER_ANCHOR, PARSER_PATCH),
    (SHORTCUT_ANCHOR, SHORTCUT_PATCH),
    (EMIT_ANCHOR, EMIT_PATCH),
):
    assert src.count(anchor) == 1, f"anchor not unique in {path}: {anchor[:60]!r}"
    src = src.replace(anchor, patch)
path.write_text(src)
print("patched", path)
