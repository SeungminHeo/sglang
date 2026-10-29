import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _parser(reasoning_effort=None, chat_template_kwargs=None, stream=True):
    request = ChatCompletionRequest(
        model="solar",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=reasoning_effort,
        chat_template_kwargs=chat_template_kwargs,
    )
    # serving_chat passes its own force_reasoning guess; the effort must win.
    return ReasoningParser(
        "solar_open2", stream_reasoning=stream, force_reasoning=True, request=request
    )


class TestSolarOpen2ReasoningParser(CustomTestCase):
    def test_effort_mirrors_template_think_prefill(self):
        """The template opens <|think:start|> only for unset/medium/high/xhigh;
        otherwise the model emits plain content, which must not be swallowed
        as truncated reasoning."""
        cases = {
            None: True,
            "medium": True,
            "high": True,
            "xhigh": True,
            "low": False,
            "none": False,
            "minimal": False,
        }
        for effort, thinking in cases.items():
            with self.subTest(effort=effort):
                reasoning, content = _parser(effort).parse_non_stream("Hello.")
                if thinking:
                    self.assertEqual((reasoning, content), ("Hello.", ""))
                else:
                    self.assertEqual((reasoning, content), ("", "Hello."))

    def test_effort_from_chat_template_kwargs(self):
        parser = _parser(chat_template_kwargs={"reasoning_effort": "low"})
        self.assertEqual(parser.parse_non_stream("Hi")[1], "Hi")

    def test_tool_call_inside_think_block_reaches_content(self):
        call = "<|tool_call:start|>f\n<|tool_call:end|>"
        reasoning, content = _parser().parse_non_stream(f"plan {call}")
        self.assertEqual(reasoning, "plan ")
        self.assertEqual(content, call)

    def test_stream_end_tag_split_across_chunks(self):
        parser = _parser()
        chunks = ["think", "ing<|th", "ink:e", "nd|>ans", "wer"]
        reasoning, content = "", ""
        for chunk in chunks:
            r, c = parser.parse_stream_chunk(chunk)
            reasoning += r or ""
            content += c or ""
        self.assertEqual((reasoning, content), ("thinking", "answer"))


class TestSolarOpen2ServingThinkingMode(CustomTestCase):
    def test_low_effort_request_applies_grammar_from_first_token(self):
        """A reasoning_effort=low request carries no think block, so serving_chat
        must not mark it as reasoning: a response_format grammar deferred to a
        think end that never comes is never applied (schema-violating JSON)."""
        serving = object.__new__(OpenAIServingChat)
        serving.reasoning_parser = "solar_open2"
        serving.template_manager = SimpleNamespace(
            reasoning_config=None, force_reasoning=False
        )
        for effort, thinking in ((None, True), ("high", True), ("low", False)):
            with self.subTest(effort=effort):
                request = ChatCompletionRequest(
                    model="solar",
                    messages=[{"role": "user", "content": "hi"}],
                    reasoning_effort=effort,
                )
                self.assertIs(serving._get_reasoning_from_request(request), thinking)


if __name__ == "__main__":
    unittest.main()
