"""Teach the Dynamo 1.6.1 SGLang chat processor about Solar Open 2's reasoning effort.

The frontend builds ReasoningParser without the request, so SolarOpen2Detector never
sees reasoning_effort and treats every response as thinking: with effort low/none
(the template prefills a closed think block) the whole answer, json_schema output
included, lands in reasoning_content and content is empty. It also resolves
force_reasoning from the static template default, which decides whether guided
decoding waits for the think block to close.

This mirrors the native SGLang server: the effort the frontend renders into the
prompt (request field first, then chat_template_kwargs, as in
_render_prompt_token_ids) decides thinking, and reaches the detector. Other parsers
are untouched. Applied at image build time; fails if the anchors moved.
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

if "_solar_open2_effort" in src:
    sys.exit(f"{path} is already patched")
for anchor, patch in (
    (IMPORT_ANCHOR, IMPORT_PATCH),
    ("\n\ndef resolve_request_force_reasoning(", EFFORT_HELPER),
    (RESOLVE_ANCHOR, RESOLVE_PATCH),
    (PARSER_ANCHOR, PARSER_PATCH),
):
    assert src.count(anchor) == 1, f"anchor not unique in {path}: {anchor[:60]!r}"
    src = src.replace(anchor, patch)
path.write_text(src)
print("patched", path)
