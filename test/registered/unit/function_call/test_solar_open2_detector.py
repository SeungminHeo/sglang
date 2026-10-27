"""Unit tests for the Solar Open 2 tool-call detector."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.solar_open2_detector import SolarOpen2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": ["integer", "null"]},
                    "ratio": {"type": "number"},
                    "exact": {"type": "boolean"},
                    "tags": {"type": "array"},
                },
            },
        ),
    ),
    Tool(type="function", function=Function(name="now", parameters={})),
]


def _call(name, **args):
    body = "".join(
        f"<|tool_arg:start|>{k}<|tool_arg:value|>{v}<|tool_arg:end|>\n"
        for k, v in args.items()
    )
    return f"<|tool_call:start|>{name}\n{body}<|tool_call:end|>"


class TestSolarOpen2Detector(CustomTestCase):
    def test_values_are_raw_strings_typed_by_schema(self):
        """Wire values are not JSON: a numeric-looking string stays a string,
        and only schema-typed params are converted."""
        text = _call(
            "search",
            query="42",
            limit="7",
            ratio="2.0",
            exact="true",
            tags='["a", "b"]',
            extra="null",
        )
        result = SolarOpen2Detector().detect_and_parse("Sure. " + text, _TOOLS)
        self.assertEqual(result.normal_text, "Sure. ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {
                "query": "42",
                "limit": 7,
                "ratio": 2,
                "exact": True,
                "tags": ["a", "b"],
                "extra": None,
            },
        )

    def test_failed_coercion_keeps_string(self):
        result = SolarOpen2Detector().detect_and_parse(
            _call("search", limit="many"), _TOOLS
        )
        self.assertEqual(json.loads(result.calls[0].parameters), {"limit": "many"})

    def test_streaming_emits_name_before_arguments_and_multiple_calls(self):
        detector = SolarOpen2Detector()
        stream = "Ok " + _call("search", query="x y") + "\n" + _call("now")
        # Token-sized pieces: each sentinel is one token.
        pieces = [
            p
            for p in stream.replace("<|", "\0<|").replace("|>", "|>\0").split("\0")
            if p
        ]
        normal, events = "", []
        for piece in pieces:
            result = detector.parse_streaming_increment(piece, _TOOLS)
            normal += result.normal_text
            events.extend((c.tool_index, c.name, c.parameters) for c in result.calls)
        self.assertEqual(normal, "Ok ")
        self.assertEqual(
            events,
            [
                (0, "search", ""),
                (0, None, '{"query": "x y"}'),
                (1, "now", ""),
                (1, None, "{}"),
            ],
        )
        self.assertEqual(
            detector.prev_tool_call_arr,
            [
                {"name": "search", "arguments": {"query": "x y"}},
                {"name": "now", "arguments": {}},
            ],
        )


if __name__ == "__main__":
    unittest.main()
