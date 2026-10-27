# Adapted from: https://github.com/UpstageAI/vllm/blob/v0.22.0-solar-open2/vllm/tool_parsers/solar_open2_tool_parser.py
import json
import logging
import re
from typing import Any, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call:start|>"
TOOL_CALL_END = "<|tool_call:end|>"
TOOL_ARG_START = "<|tool_arg:start|>"
TOOL_ARG_VALUE = "<|tool_arg:value|>"
TOOL_ARG_END = "<|tool_arg:end|>"

_TOOL_CALL_RE = re.compile(
    rf"{re.escape(TOOL_CALL_START)}(.+?)\n"
    rf"((?:{re.escape(TOOL_ARG_START)}.*?{re.escape(TOOL_ARG_END)}\n?)*)"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)
_TOOL_ARG_RE = re.compile(
    rf"{re.escape(TOOL_ARG_START)}(.*?){re.escape(TOOL_ARG_VALUE)}(.*?)"
    rf"{re.escape(TOOL_ARG_END)}",
    re.DOTALL,
)


def _param_type(func_name: str, param_name: str, tools: List[Tool]) -> str:
    """JSON-schema type of the param; the first non-null member of a type
    union; ``"string"`` when the schema says nothing."""
    for tool in tools:
        if tool.function.name != func_name:
            continue
        params = tool.function.parameters
        props = params.get("properties") if isinstance(params, dict) else None
        spec = props.get(param_name) if isinstance(props, dict) else None
        if not isinstance(spec, dict):
            return "string"
        param_type = spec.get("type", "string")
        if isinstance(param_type, list):
            return next((str(t) for t in param_type if t != "null"), "string")
        return str(param_type)
    return "string"


def _coerce(value: str, param_type: str) -> Any:
    """Convert a raw wire-format value to its schema type; on failure keep the
    string so malformed output still reaches the client."""
    if value.strip().lower() == "null":
        return None
    pt = param_type.strip().lower()
    try:
        if pt in ("integer", "int"):
            return int(value)
        if pt in ("number", "float", "double"):
            number = float(value)
            # Integral numbers round-trip as int, as type-strict evaluators expect.
            return int(number) if number - int(number) == 0 else number
        if pt in ("boolean", "bool"):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
            return value
        if pt in ("array", "list", "object", "dict"):
            return json.loads(value)
    except (ValueError, TypeError, OverflowError):
        logger.warning(
            "solar_open2: could not coerce %r to %s; keeping the string.",
            value,
            param_type,
        )
        return value
    if pt in ("null", "none"):
        return None
    return value


def _parse_arguments(func_name: str, args_block: str, tools: List[Tool]) -> dict:
    return {
        name: _coerce(value, _param_type(func_name, name, tools))
        for name, value in _TOOL_ARG_RE.findall(args_block)
    }


class SolarOpen2Detector(BaseFormatDetector):
    """
    Detector for Solar Open 2 (upstage/Solar-Open2-250B).

    Format (argument values are raw strings, typed from the tool schema):
      <|tool_call:start|>get_weather
      <|tool_arg:start|>city<|tool_arg:value|>Seoul<|tool_arg:end|>
      <|tool_call:end|>

    Text after the first tool call is dropped, matching the reference parser:
    the chat template forbids text after a call.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_CALL_START
        self.eot_token = TOOL_CALL_END
        self._in_tool_section = False

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text)
        calls = []
        for match in _TOOL_CALL_RE.finditer(text):
            func_name = match.group(1).strip()
            arguments = _parse_arguments(func_name, match.group(2), tools)
            calls.extend(
                self.parse_base_json(
                    {"name": func_name, "parameters": arguments}, tools
                )
            )
        return StreamingParseResult(normal_text=text[:idx], calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        if not self._in_tool_section:
            idx = self._buffer.find(self.bot_token)
            if idx == -1:
                held = self._ends_with_partial_token(self._buffer, self.bot_token)
                normal_text = self._buffer[: len(self._buffer) - held]
                self._buffer = self._buffer[len(self._buffer) - held :]
                return StreamingParseResult(normal_text=normal_text)
            normal_text = self._buffer[:idx]
            self._buffer = self._buffer[idx:]
            self._in_tool_section = True
        else:
            normal_text = ""
        return StreamingParseResult(
            normal_text=normal_text, calls=self._drain_tool_calls(tools)
        )

    def _drain_tool_calls(self, tools: List[Tool]) -> List[ToolCallItem]:
        """Emit each call's name once its line is complete and its arguments
        once the call closes; anything between calls is discarded."""
        calls: List[ToolCallItem] = []
        while True:
            start = self._buffer.find(self.bot_token)
            if start == -1:
                self._buffer = self._buffer[
                    len(self._buffer)
                    - self._ends_with_partial_token(self._buffer, self.bot_token) :
                ]
                return calls
            if not self.current_tool_name_sent:
                name_end = self._buffer.find("\n", start)
                if name_end == -1:
                    return calls
                func_name = self._buffer[start + len(self.bot_token) : name_end]
                self._start_tool_call(func_name.strip(), calls)
            match = _TOOL_CALL_RE.match(self._buffer, start)
            if match is None:
                return calls
            self._finish_tool_call(match.group(2), tools, calls)
            self._buffer = self._buffer[match.end() :]

    def _start_tool_call(self, func_name: str, calls: List[ToolCallItem]) -> None:
        self.current_tool_id += 1
        self.current_tool_name_sent = True
        self.prev_tool_call_arr.append({"name": func_name, "arguments": {}})
        self.streamed_args_for_tool.append("")
        calls.append(
            ToolCallItem(tool_index=self.current_tool_id, name=func_name, parameters="")
        )

    def _finish_tool_call(
        self, args_block: str, tools: List[Tool], calls: List[ToolCallItem]
    ) -> None:
        tool_call = self.prev_tool_call_arr[self.current_tool_id]
        tool_call["arguments"] = _parse_arguments(tool_call["name"], args_block, tools)
        arguments_json = json.dumps(tool_call["arguments"], ensure_ascii=False)
        self.streamed_args_for_tool[self.current_tool_id] = arguments_json
        self.current_tool_name_sent = False
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id, name=None, parameters=arguments_json
            )
        )

    def supports_structural_tag(self) -> bool:
        # Arguments are not JSON, so JSON-schema structural tags cannot express them.
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()


__all__ = ["SolarOpen2Detector"]
