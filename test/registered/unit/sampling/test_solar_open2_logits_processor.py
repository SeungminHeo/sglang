# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from UpstageAI/vllm v0.22.0-solar-open2 tests/v1/logits_processors/test_solar_open2.py
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.custom_logit_processor import (
    CustomLogitProcessor,
    should_delay_sampling,
)
from sglang.srt.sampling.solar_open2 import (
    DISABLE_EXTRA_ARG,
    REASONING_BUDGET_EXTRA_ARG,
    SolarOpen2ProcessorDefaults,
    SolarOpen2TemplateLogitProcessor,
    SolarOpen2TokenFSMEnforcer,
    SolarOpen2TokenIds,
    _normalize_reasoning_budget,
    _resolve_token_ids_from_tokenizer_dirs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


IDS = SolarOpen2TokenIds(
    im_start=128,
    im_end=129,
    think_start=130,
    think_end=131,
    im_content=132,
    tool_start=133,
    tool_end=134,
    tool_call_start=135,
    tool_call_end=136,
    tool_arg_start=137,
    tool_arg_value=138,
    tool_arg_end=139,
    tool_response_start=140,
    tool_response_end=141,
)


REGULAR_TOKEN = 99


VOCAB_SIZE = 160


def _logits() -> torch.Tensor:
    return torch.arange(VOCAB_SIZE, dtype=torch.float32)


def _apply(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    has_structured_outputs: bool = False,
) -> torch.Tensor:
    enforcer = SolarOpen2TokenFSMEnforcer(
        prompt_token_ids,
        IDS,
        has_structured_outputs=has_structured_outputs,
    )
    return enforcer(prompt_token_ids, output_token_ids, _logits())


def _is_masked(logits: torch.Tensor, token_id: int | None) -> bool:
    assert token_id is not None
    return logits[token_id].item() == float("-inf")


def _is_available(logits: torch.Tensor, token_id: int | None) -> bool:
    assert token_id is not None
    return logits[token_id].item() != float("-inf")


def test_open_reasoning_prompt_allows_only_think_end_special():
    logits = _apply([IDS.think_start], [])

    assert _is_available(logits, IDS.think_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_end)
    assert _is_masked(logits, IDS.think_start)
    assert _is_masked(logits, IDS.tool_call_start)
    assert _is_masked(logits, IDS.tool_arg_start)
    assert _is_masked(logits, IDS.tool_response_start)


def test_empty_reasoning_transitions_to_content_state():
    logits = _apply([IDS.think_start], [IDS.think_end])

    assert _is_available(logits, IDS.tool_call_start)
    # Fresh CONTENT (no content token yet): ending the turn is forbidden
    # until the model produces at least one content token or a tool call.
    assert _is_masked(logits, IDS.im_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_start)
    assert _is_masked(logits, IDS.im_content)
    assert _is_masked(logits, IDS.think_start)
    assert _is_masked(logits, IDS.think_end)
    assert _is_masked(logits, IDS.tool_start)
    assert _is_masked(logits, IDS.tool_end)
    assert _is_masked(logits, IDS.tool_arg_value)


def test_prompt_with_closed_think_block_starts_in_content_state():
    logits = _apply([IDS.think_start, IDS.think_end], [])

    assert _is_available(logits, IDS.tool_call_start)
    # Empty-content guard applies from the start for prompt-closed pairs.
    assert _is_masked(logits, IDS.im_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.think_start)
    assert _is_masked(logits, IDS.think_end)


def test_structured_outputs_do_not_mask_content_logits():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [],
        has_structured_outputs=True,
    )

    for token_id in (
        IDS.think_start,
        IDS.think_end,
        IDS.im_start,
        IDS.im_end,
        IDS.tool_call_start,
        IDS.tool_arg_value,
        REGULAR_TOKEN,
    ):
        assert _is_available(logits, token_id)


def test_tool_call_begin_requires_a_regular_function_name_token():
    logits = _apply([IDS.think_start, IDS.think_end], [IDS.tool_call_start])

    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_end)
    assert _is_masked(logits, IDS.tool_arg_start)
    assert _is_masked(logits, IDS.tool_call_end)


def test_tool_call_name_allows_args_or_call_end_specials():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [IDS.tool_call_start, REGULAR_TOKEN],
    )

    assert _is_available(logits, IDS.tool_arg_start)
    assert _is_available(logits, IDS.tool_call_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.tool_arg_value)
    assert _is_masked(logits, IDS.think_end)


def test_tool_arg_name_allows_only_value_marker_special():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_arg_start, REGULAR_TOKEN],
    )

    assert _is_available(logits, IDS.tool_arg_value)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.tool_arg_start)
    assert _is_masked(logits, IDS.tool_call_end)


def test_tool_arg_end_allows_separator_text_or_next_marker():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [
            IDS.tool_call_start,
            REGULAR_TOKEN,
            IDS.tool_arg_start,
            REGULAR_TOKEN,
            IDS.tool_arg_value,
            REGULAR_TOKEN,
            IDS.tool_arg_end,
        ],
    )

    assert _is_available(logits, IDS.tool_arg_start)
    assert _is_available(logits, IDS.tool_call_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.tool_arg_value)


def test_tool_call_end_allows_regular_stop_text_or_another_tool_call():
    logits = _apply(
        [IDS.think_start, IDS.think_end],
        [IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_call_end],
    )

    assert _is_available(logits, IDS.tool_call_start)
    assert _is_available(logits, IDS.im_end)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.tool_arg_start)
    assert _is_masked(logits, IDS.think_end)
    assert _is_masked(logits, IDS.tool_response_end)


def test_incremental_state_resets_to_prompt_state_when_output_shrinks():
    enforcer = SolarOpen2TokenFSMEnforcer([IDS.think_start], IDS)

    logits = enforcer([IDS.think_start], [IDS.think_end], _logits())
    assert _is_masked(logits, IDS.think_end)
    logits = enforcer([IDS.think_start], [], _logits())

    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, IDS.tool_call_start)


def test_tokenizer_json_resolution_reads_added_tokens(tmp_path: Path):
    (tmp_path / "tokenizer_config.json").write_text(
        """
        {
          "added_tokens_decoder": {
            "128": {"content": "<|im:start|>"},
            "129": {"content": "<|im:end|>"},
            "130": {"content": "<|think:start|>"},
            "131": {"content": "<|think:end|>"},
            "132": {"content": "<|im:content|>"},
            "133": {"content": "<|tool:start|>"},
            "134": {"content": "<|tool:end|>"},
            "135": {"content": "<|tool_call:start|>"}
          }
        }
        """,
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text(
        """
        {
          "added_tokens": [
            {"id": 136, "content": "<|tool_call:end|>"},
            {"id": 137, "content": "<|tool_arg:start|>"},
            {"id": 138, "content": "<|tool_arg:value|>"},
            {"id": 139, "content": "<|tool_arg:end|>"},
            {"id": 140, "content": "<|tool_response:start|>"},
            {"id": 141, "content": "<|tool_response:end|>"}
          ]
        }
        """,
        encoding="utf-8",
    )

    assert _resolve_token_ids_from_tokenizer_dirs([tmp_path]) == {
        "im_start": 128,
        "im_end": 129,
        "think_start": 130,
        "think_end": 131,
        "im_content": 132,
        "tool_start": 133,
        "tool_end": 134,
        "tool_call_start": 135,
        "tool_call_end": 136,
        "tool_arg_start": 137,
        "tool_arg_value": 138,
        "tool_arg_end": 139,
        "tool_response_start": 140,
        "tool_response_end": 141,
    }


def _validate(extra_args):
    SolarOpen2TemplateLogitProcessor.validate_params(extra_args)


def test_validate_params_accepts_integer_disable_flag():
    # The per-request disable flag is an integer (non-zero disables; 0 or
    # omitted keeps the processor on). int transports natively over the HTTP
    # vllm_xargs field. None / {} / 0 / non-zero / bool are all valid, and an
    # unrelated extra_arg must not trip the disable validation.
    _validate(None)
    _validate({})
    _validate({DISABLE_EXTRA_ARG: 0})
    _validate({DISABLE_EXTRA_ARG: 1})
    _validate({DISABLE_EXTRA_ARG: True})
    _validate({"other": "x"})


def test_validate_params_rejects_non_integer_disable_flag():
    for bad in ("1", 1.5, [1]):
        with pytest.raises(ValueError):
            _validate({DISABLE_EXTRA_ARG: bad})


EOS_ID = 2  # e.g. <|endoftext|> — listed in generation_config eos_token_id


def _apply_with_eos(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
) -> torch.Tensor:
    enforcer = SolarOpen2TokenFSMEnforcer(
        prompt_token_ids,
        IDS,
        extra_eos_token_ids=frozenset({EOS_ID}),
    )
    return enforcer(prompt_token_ids, output_token_ids, _logits())


def test_reasoning_masks_non_sentinel_eos():
    # Open think block in prompt -> REASONING: the model must not be able to
    # end its turn before emitting <|think:end|>.
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], [])
    assert _is_masked(logits, EOS_ID)
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, IDS.im_end)


def test_content_allows_non_sentinel_eos():
    # CONTENT with at least one content token -> the turn may legally end,
    # so EOS stays available (no behavior change for successful turns).
    logits = _apply_with_eos(
        [IDS.im_content, IDS.think_start], [IDS.think_end, REGULAR_TOKEN]
    )
    assert _is_available(logits, EOS_ID)
    assert _is_available(logits, IDS.im_end)


def test_tool_call_states_mask_non_sentinel_eos():
    # Mid tool-call: ending the turn would leave an unterminated call.
    output = [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_arg_start]
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], output)
    assert _is_masked(logits, EOS_ID)


def test_tool_call_end_allows_non_sentinel_eos():
    output = [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_call_end]
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], output)
    assert _is_available(logits, EOS_ID)
    assert _is_available(logits, IDS.im_end)


def test_no_extra_eos_preserves_legacy_behavior():
    # Without resolved extra EOS ids the enforcer behaves exactly as before.
    logits = _apply([IDS.im_content, IDS.think_start], [])
    assert _is_available(logits, EOS_ID)


def test_fresh_content_masks_turn_end_until_progress():
    # Right after <|think:end|>: no content yet -> both <|im:end|> and bare
    # EOS are masked; regular tokens and tool calls remain available.
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], [IDS.think_end])
    assert _is_masked(logits, IDS.im_end)
    assert _is_masked(logits, EOS_ID)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.tool_call_start)


def test_content_progress_reenables_turn_end():
    logits = _apply_with_eos(
        [IDS.im_content, IDS.think_start], [IDS.think_end, REGULAR_TOKEN]
    )
    assert _is_available(logits, IDS.im_end)
    assert _is_available(logits, EOS_ID)


def test_completed_tool_call_counts_as_content_progress():
    # think_end -> tool call -> back in CONTENT: the completed call is
    # sufficient content, so the turn may end without extra text.
    output = [
        IDS.think_end,
        IDS.tool_call_start,
        REGULAR_TOKEN,
        IDS.tool_call_end,
        REGULAR_TOKEN,
    ]
    logits = _apply_with_eos([IDS.im_content, IDS.think_start], output)
    assert _is_available(logits, IDS.im_end)
    assert _is_available(logits, EOS_ID)


_BATCH_SCENARIOS: list[tuple[list[int], list[int], bool]] = [
    # (prompt, output, has_structured_outputs)
    ([IDS.think_start], [], False),  # REASONING
    ([IDS.think_start], [IDS.think_end], False),  # CONTENT fresh
    ([IDS.think_start], [IDS.think_end, REGULAR_TOKEN], False),  # CONTENT progress
    ([IDS.think_start], [IDS.think_end, IDS.tool_call_start], False),
    (
        [IDS.think_start],
        [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_arg_start],
        False,
    ),
    (
        [IDS.think_start],
        [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_call_end],
        False,
    ),
    ([IDS.think_start], [IDS.think_end, REGULAR_TOKEN], True),  # structured: no mask
    ([IDS.think_start], [], True),  # structured but REASONING: masked
]


def _batch_logits(num_rows: int) -> torch.Tensor:
    return torch.arange(num_rows * VOCAB_SIZE, dtype=torch.float32).reshape(
        num_rows, VOCAB_SIZE
    )


def _apply_budget(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    reasoning_budget: int | None,
    extra_eos: bool = False,
) -> torch.Tensor:
    enforcer = SolarOpen2TokenFSMEnforcer(
        prompt_token_ids,
        IDS,
        reasoning_budget=reasoning_budget,
        extra_eos_token_ids=frozenset({EOS_ID}) if extra_eos else frozenset(),
    )
    return enforcer(prompt_token_ids, output_token_ids, _logits())


def test_reasoning_budget_below_cap_allows_normal_reasoning():
    # 2 of a 3-token budget consumed: regular reasoning text and <|think:end|>
    # remain available; nothing is force-emitted yet.
    logits = _apply_budget(
        [IDS.think_start], [REGULAR_TOKEN, REGULAR_TOKEN], reasoning_budget=3
    )
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, IDS.im_end)


def test_reasoning_budget_at_cap_forces_think_end():
    # 3rd of a 3-token budget consumed: the next step must be <|think:end|>;
    # every other token (regular text and all sentinels) is masked.
    logits = _apply_budget(
        [IDS.think_start],
        [REGULAR_TOKEN, REGULAR_TOKEN, REGULAR_TOKEN],
        reasoning_budget=3,
    )
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_end)
    assert _is_masked(logits, IDS.tool_call_start)


def test_reasoning_budget_one_forces_after_single_token():
    logits = _apply_budget([IDS.think_start], [REGULAR_TOKEN], reasoning_budget=1)
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, REGULAR_TOKEN)


def test_reasoning_budget_none_is_unbounded():
    # Legacy behavior: without a budget, reasoning text stays available no
    # matter how many reasoning tokens have been emitted.
    logits = _apply_budget(
        [IDS.think_start], [REGULAR_TOKEN] * 50, reasoning_budget=None
    )
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)


def test_reasoning_budget_zero_disables_cap():
    # 0 normalizes to "no cap" so callers can opt out of a server default.
    logits = _apply_budget([IDS.think_start], [REGULAR_TOKEN] * 5, reasoning_budget=0)
    assert _is_available(logits, REGULAR_TOKEN)


def test_reasoning_budget_does_not_apply_in_content():
    # After <|think:end|> the FSM is in CONTENT; the budget never force-closes
    # there. Regular content tokens stay available (fresh-content guard still
    # masks <|im:end|> until progress, independent of the budget).
    logits = _apply_budget(
        [IDS.think_start], [REGULAR_TOKEN, IDS.think_end], reasoning_budget=1
    )
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_masked(logits, IDS.im_end)


def test_reasoning_budget_resets_when_output_shrinks():
    # Recomputation (output shorter than last seen) resets the counter to the
    # prompt state, so the budget is re-evaluated from scratch.
    enforcer = SolarOpen2TokenFSMEnforcer([IDS.think_start], IDS, reasoning_budget=3)
    forced = enforcer(
        [IDS.think_start],
        [REGULAR_TOKEN, REGULAR_TOKEN, REGULAR_TOKEN],
        _logits(),
    )
    assert _is_masked(forced, REGULAR_TOKEN)  # at cap

    rewound = enforcer([IDS.think_start], [REGULAR_TOKEN], _logits())
    assert _is_available(rewound, REGULAR_TOKEN)  # counter reset, 1 < 3


def test_reasoning_budget_restarts_on_new_think_block():
    # A fresh <|think:start|> resets the counter mid-turn. Here the prompt opens
    # a (later closed) block; the model reopens reasoning, and the budget counts
    # only tokens of the new block.
    logits = _apply_budget(
        [IDS.think_start, IDS.think_end],
        [REGULAR_TOKEN, IDS.think_start, REGULAR_TOKEN],
        reasoning_budget=2,
    )
    # New block has 1 of 2 tokens -> still below cap.
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)


def test_reasoning_budget_then_empty_content_guard_chain():
    # End-to-end: budget truncates reasoning, the model emits the forced
    # <|think:end|>, and the empty-content guard then forbids ending the turn
    # until real content appears (both <|im:end|> and bare EOS masked).
    forced = _apply_budget(
        [IDS.im_content, IDS.think_start],
        [REGULAR_TOKEN, REGULAR_TOKEN],
        reasoning_budget=2,
        extra_eos=True,
    )
    assert _is_available(forced, IDS.think_end)
    assert _is_masked(forced, REGULAR_TOKEN)

    after_close = _apply_budget(
        [IDS.im_content, IDS.think_start],
        [REGULAR_TOKEN, REGULAR_TOKEN, IDS.think_end],
        reasoning_budget=2,
        extra_eos=True,
    )
    assert _is_masked(after_close, IDS.im_end)
    assert _is_masked(after_close, EOS_ID)
    assert _is_available(after_close, REGULAR_TOKEN)


def test_reasoning_budget_unenforceable_without_think_end_id():
    # If <|think:end|> is unresolved we cannot force a close; the budget is a
    # no-op rather than masking the whole vocab into a dead end.
    ids_no_end = SolarOpen2TokenIds(think_start=130, think_end=None, im_end=129)
    enforcer = SolarOpen2TokenFSMEnforcer([130], ids_no_end, reasoning_budget=1)
    logits = enforcer([130], [REGULAR_TOKEN, REGULAR_TOKEN], _logits())
    assert _is_available(logits, REGULAR_TOKEN)


def test_reasoning_budget_with_structured_outputs_still_caps_reasoning():
    # Structured outputs only bypass masking in CONTENT; reasoning is still
    # FSM-governed, so the budget force-closes <|think:end|> as usual.
    enforcer = SolarOpen2TokenFSMEnforcer(
        [IDS.think_start], IDS, has_structured_outputs=True, reasoning_budget=2
    )
    logits = enforcer([IDS.think_start], [REGULAR_TOKEN, REGULAR_TOKEN], _logits())
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, REGULAR_TOKEN)


def test_normalize_reasoning_budget_values():
    assert _normalize_reasoning_budget(None) is None
    assert _normalize_reasoning_budget(0) is None
    assert _normalize_reasoning_budget(5) == 5
    for bad in (True, False, "5", 1.5, [1]):
        with pytest.raises(ValueError):
            _normalize_reasoning_budget(bad)
    with pytest.raises(ValueError):
        _normalize_reasoning_budget(-1)


def test_validate_params_accepts_and_rejects_reasoning_budget():
    SolarOpen2TemplateLogitProcessor.validate_params({REASONING_BUDGET_EXTRA_ARG: 0})
    SolarOpen2TemplateLogitProcessor.validate_params({REASONING_BUDGET_EXTRA_ARG: 4096})
    for bad in ("4096", 1.5, -1, True):
        with pytest.raises(ValueError):
            SolarOpen2TemplateLogitProcessor.validate_params(
                {REASONING_BUDGET_EXTRA_ARG: bad}
            )


NEWLINE_ID = 150  # stand-in for the "\n\n" token id (< VOCAB_SIZE, not a sentinel)


def _apply_leading(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    leading_ids: frozenset[int] = frozenset({NEWLINE_ID}),
) -> torch.Tensor:
    enforcer = SolarOpen2TokenFSMEnforcer(
        prompt_token_ids,
        IDS,
        leading_forbidden_ids=leading_ids,
    )
    return enforcer(prompt_token_ids, output_token_ids, _logits())


def test_leading_newline_masked_right_after_think_start():
    # Prompt ends with <|think:start|> -> the first generated token is the
    # leading reasoning token: the newline is masked while ordinary reasoning
    # text and <|think:end|> stay available (normal REASONING mask preserved).
    logits = _apply_leading([IDS.think_start], [])
    assert _is_masked(logits, NEWLINE_ID)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)
    assert _is_masked(logits, IDS.im_end)


def test_leading_newline_allowed_after_first_reasoning_token():
    # One reasoning token already emitted -> past the leading position, so the
    # newline is available again (only the *leading* newline triggers the leak).
    logits = _apply_leading([IDS.think_start], [REGULAR_TOKEN])
    assert _is_available(logits, NEWLINE_ID)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)


def test_leading_newline_masked_when_model_emits_think_start():
    # <|think:start|> generated mid-stream (not in the prompt): the token right
    # after it is still the leading position.
    logits = _apply_leading([IDS.im_content], [IDS.think_start])
    assert _is_masked(logits, NEWLINE_ID)
    assert _is_available(logits, REGULAR_TOKEN)


def test_leading_newline_not_masked_mid_prompt_reasoning():
    # Prompt opens reasoning AND already contains a reasoning token: the next
    # token is a continuation, not the leading position -> newline available.
    logits = _apply_leading([IDS.think_start, REGULAR_TOKEN], [])
    assert _is_available(logits, NEWLINE_ID)


def test_leading_newline_no_ban_without_config():
    # Default (no leading ids configured): behavior is unchanged, so the newline
    # stays available even right after <|think:start|>.
    logits = _apply_leading([IDS.think_start], [], leading_ids=frozenset())
    assert _is_available(logits, NEWLINE_ID)


def test_leading_newline_reset_on_output_shrink():
    # Rewind (output shorter than last seen) restores the leading flag from the
    # prompt so the ban re-applies on recomputation.
    enforcer = SolarOpen2TokenFSMEnforcer(
        [IDS.think_start], IDS, leading_forbidden_ids=frozenset({NEWLINE_ID})
    )
    advanced = enforcer([IDS.think_start], [REGULAR_TOKEN], _logits())
    assert _is_available(advanced, NEWLINE_ID)  # past the leading position
    rewound = enforcer([IDS.think_start], [], _logits())
    assert _is_masked(rewound, NEWLINE_ID)  # leading position restored


def test_leading_newline_bypass_closed_with_single_and_double():
    # Banning both the single ("\n") and double ("\n\n") newline ids at the
    # leading position closes the two-token bypass: the model can neither emit
    # "\n\n" directly nor start a "\n" + "\n" run.
    single_nl, double_nl = 150, 151
    logits = _apply_leading(
        [IDS.think_start], [], leading_ids=frozenset({single_nl, double_nl})
    )
    assert _is_masked(logits, single_nl)
    assert _is_masked(logits, double_nl)
    assert _is_available(logits, REGULAR_TOKEN)
    assert _is_available(logits, IDS.think_end)


def _request(prompt=None, output=None, structured=False, params=None):
    sampling = SimpleNamespace(
        custom_params=params or {},
        json_schema="{}" if structured else None,
        regex=None,
        ebnf=None,
        structural_tag=None,
    )
    req = SimpleNamespace(
        origin_input_ids=prompt if prompt is not None else [IDS.think_start],
        output_ids=output if output is not None else [],
        sampling_params=sampling,
    )
    return req, {**sampling.custom_params, "__req__": req}


@pytest.mark.parametrize("budget", [None, 0, 1, 3])
def test_sglang_batch_matches_upstage_scalar(budget):
    scenarios = _BATCH_SCENARIOS + [
        ([IDS.think_start], [REGULAR_TOKEN] * 3, False),
        (
            [IDS.think_start],
            [
                IDS.think_end,
                IDS.tool_call_start,
                REGULAR_TOKEN,
                IDS.tool_arg_start,
                REGULAR_TOKEN,
                IDS.tool_arg_value,
            ],
            False,
        ),
        (
            [IDS.think_start],
            [
                IDS.think_end,
                IDS.tool_call_start,
                REGULAR_TOKEN,
                IDS.tool_arg_start,
                REGULAR_TOKEN,
                IDS.tool_arg_value,
                REGULAR_TOKEN,
                IDS.tool_arg_end,
            ],
            False,
        ),
    ]
    params = [_request(p, list(o), s)[1] for p, o, s in scenarios]
    proc = SolarOpen2TemplateLogitProcessor(
        default_reasoning_budget=budget, leading_forbidden_ids=frozenset({NEWLINE_ID})
    )
    actual = proc(_batch_logits(len(params)), params)
    expected = _batch_logits(len(params))
    for row, (p, o, s) in enumerate(scenarios):
        fsm = SolarOpen2TokenFSMEnforcer(
            p, IDS, s, frozenset({2}), budget, frozenset({NEWLINE_ID})
        )
        fsm(p, o, expected[row])
    assert torch.equal(actual, expected)


def test_sglang_state_survives_processor_reconstruction_and_row_reorder():
    a, pa = _request()
    b, pb = _request(output=[IDS.think_end, REGULAR_TOKEN])
    serial = SolarOpen2TemplateLogitProcessor().serialize()
    CustomLogitProcessor.from_str(serial)(_batch_logits(2), [pa, pb])
    state = a._solar_open2_enforcer
    a.output_ids.extend(
        [IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN, IDS.tool_call_end]
    )
    result = CustomLogitProcessor.from_str(serial)(_batch_logits(2), [pb, pa])
    assert a._solar_open2_enforcer is state
    assert state._last_processed_len == 4
    assert _is_available(result[1], IDS.im_end)
    result = CustomLogitProcessor.from_str(serial)(_batch_logits(2), [pb, pa])
    assert state._last_processed_len == 4
    a.output_ids.clear()
    result = CustomLogitProcessor.from_str(serial)(_batch_logits(1), [pa])
    assert state._last_processed_len == 0
    assert _is_masked(result[0], IDS.im_end)


def test_overlap_commits_tool_call_end_before_delayed_sampling():
    req, params = _request(output=[IDS.think_end, IDS.tool_call_start, REGULAR_TOKEN])
    proc = SolarOpen2TemplateLogitProcessor()
    # Previous GPU step produced tool_call:end, but output_ids is still stale.
    stale = proc(_batch_logits(1), [params])
    assert _is_masked(stale[0], IDS.im_end)
    info = SimpleNamespace(grammars=None, has_custom_logit_processor=True)
    assert should_delay_sampling(
        enable_overlap=True,
        enable_spec=False,
        sampling_info=info,
        is_prefill_only=False,
        force_delay=False,
    )
    delayed = lambda: proc(_batch_logits(1), [params])
    req.output_ids.append(IDS.tool_call_end)  # scheduler.process_batch_result
    fresh = delayed()  # scheduler.launch_batch_sample_if_needed
    assert _is_available(fresh[0], IDS.im_end)
    assert _is_available(fresh[0], EOS_ID)


@pytest.mark.parametrize("overlap", [False, True])
def test_pd_prefill_first_token_forbids_newline(overlap):
    req, params = _request()
    proc = SolarOpen2TemplateLogitProcessor(
        leading_forbidden_ids=frozenset({NEWLINE_ID})
    )
    info = SimpleNamespace(grammars=None, has_custom_logit_processor=True)
    # PD prefill has max_new_tokens=1, unlike zero-token prefill-only requests.
    assert (
        should_delay_sampling(
            enable_overlap=overlap,
            enable_spec=False,
            sampling_info=info,
            is_prefill_only=False,
            force_delay=False,
        )
        == overlap
    )
    out = proc(_batch_logits(1), [params])
    assert _is_masked(out[0], NEWLINE_ID)
    req.output_ids.append(REGULAR_TOKEN)
    assert _is_available(proc(_batch_logits(1), [params])[0], NEWLINE_ID)
    assert not should_delay_sampling(
        enable_overlap=True,
        enable_spec=False,
        sampling_info=info,
        is_prefill_only=True,
        force_delay=False,
    )


@pytest.mark.parametrize(
    "constraint", ["json_schema", "regex", "ebnf", "structural_tag"]
)
def test_structured_content_is_untouched(constraint):
    req, params = _request(output=[IDS.think_end])
    setattr(req.sampling_params, constraint, "constraint")
    logits = _batch_logits(1)
    assert torch.equal(
        SolarOpen2TemplateLogitProcessor()(logits.clone(), [params]), logits
    )


def test_request_budget_and_disable():
    req, params = _request(
        output=[REGULAR_TOKEN], params={REASONING_BUDGET_EXTRA_ARG: 1}
    )
    out = SolarOpen2TemplateLogitProcessor()(_batch_logits(1), [params])
    assert torch.isfinite(out).sum() == 1
    assert _is_available(out[0], IDS.think_end)
    _, unlimited = _request(
        output=[REGULAR_TOKEN] * 3, params={REASONING_BUDGET_EXTRA_ARG: 0}
    )
    out = SolarOpen2TemplateLogitProcessor(default_reasoning_budget=1)(
        _batch_logits(1), [unlimited]
    )
    assert _is_available(out[0], REGULAR_TOKEN)
    _, disabled = _request(params={DISABLE_EXTRA_ARG: 1})
    assert torch.equal(
        SolarOpen2TemplateLogitProcessor()(_batch_logits(1), [disabled]),
        _batch_logits(1),
    )


def _defaults(
    tmp_path,
    *,
    disabled=False,
    architecture="SolarOpen2ForCausalLM",
    spec=None,
    budget=None,
    tokenizer=None,
):
    return SolarOpen2ProcessorDefaults(
        model_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        disabled=disabled,
        reasoning_budget=budget,
        speculative_algorithm=spec,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture], eos_token_id=2),
            hf_generation_config=SimpleNamespace(eos_token_id=[2, 129]),
        ),
        tokenizer=tokenizer,
    )


def test_default_attachment_and_client_conflict(tmp_path):
    defaults = _defaults(tmp_path)
    sampling = SimpleNamespace(custom_params=None)
    serial = defaults.for_request(None, sampling)
    assert sampling.custom_params == {}
    assert isinstance(
        CustomLogitProcessor.from_str(serial), SolarOpen2TemplateLogitProcessor
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        defaults.for_request("client", sampling)
    sampling.custom_params = {DISABLE_EXTRA_ARG: 1}
    assert defaults.for_request("client", sampling) == "client"
    assert (
        _defaults(tmp_path, disabled=True).for_request("client", sampling) == "client"
    )
    assert _defaults(tmp_path, architecture="Other").serialized is None
    with pytest.raises(ValueError, match="speculative"):
        _defaults(tmp_path, spec="EAGLE")
    with pytest.raises(ValueError):
        _defaults(tmp_path, budget=-1)
    sampling.custom_params = {REASONING_BUDGET_EXTRA_ARG: "1"}
    with pytest.raises(ValueError):
        defaults.for_request(None, sampling)


def test_metadata_resolution_and_serialization(tmp_path):
    import json

    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {
                        "<|think:start|>": 10,
                        "<|think:end|>": 11,
                        "Ċ": 12,
                        "ĊĊ": 13,
                        "Ċtext": 14,
                    }
                }
            }
        )
    )
    p = CustomLogitProcessor.from_str(_defaults(tmp_path).serialized)
    assert p.token_ids.think_start == 10
    assert p.token_ids.think_end == 11
    assert p.token_ids.im_end == 129
    assert p.leading_forbidden_ids == frozenset({12, 13})
    assert p.extra_eos_token_ids == frozenset({2})
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"<|think:start|>": 15, "ĊĊĊ": 16}, eos_token_id=3
    )
    p = CustomLogitProcessor.from_str(
        _defaults(tmp_path, tokenizer=tokenizer).serialized
    )
    assert p.token_ids.think_start == 15
    assert p.leading_forbidden_ids == frozenset({12, 13, 16})
    assert p.extra_eos_token_ids == frozenset({2, 3})


def test_batch_attachment_without_client_code_permission_and_spec_rejection(request):
    from unittest.mock import MagicMock

    from sglang.srt.layers.sampler import apply_custom_logit_processor
    from sglang.srt.runtime_context import get_context
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_params import SamplingParams

    override = get_context().override_server_args(enable_custom_logit_processor=False)
    override.install()
    request.addfinalizer(override.restore)
    req, params = _request()
    req.sampling_params = SamplingParams(custom_params=params, top_k=1)
    req.custom_logit_processor = SolarOpen2TemplateLogitProcessor().serialize()
    req.return_sampling_mask = False
    req.sampling_logprobs_mode = "selected"
    req.origin_input_ids = [IDS.think_start]
    req.tokenizer = SimpleNamespace(eos_token_id=2, additional_stop_token_ids=None)
    batch = MagicMock()
    batch.reqs, batch.device = [req], "cpu"
    info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
    assert info.has_custom_logit_processor
    logits = _batch_logits(1)
    apply_custom_logit_processor(logits, info)
    assert _is_masked(logits[0], IDS.im_end)
    with pytest.raises(ValueError, match="speculative"):
        apply_custom_logit_processor(_batch_logits(2), info, num_tokens_in_batch=2)
    # Filtering and merging retain the built-in through the normal SGLang path.
    other = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
    info.merge_batch(other)
    info.filter_batch([1], torch.tensor([1]))
    assert info.has_custom_logit_processor
    assert next(iter(info.custom_logit_processor.values())).rows == [0]
    logits = _batch_logits(1)
    req.output_ids.extend([IDS.think_end, REGULAR_TOKEN])
    apply_custom_logit_processor(logits, info)
    assert _is_available(logits[0], IDS.im_end)


def test_disabled_client_processor_in_mixed_batch_is_skipped_not_raised(request):
    from unittest.mock import MagicMock

    from sglang.srt.runtime_context import get_context
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_params import SamplingParams

    override = get_context().override_server_args(enable_custom_logit_processor=False)
    override.install()
    request.addfinalizer(override.restore)
    reqs = []
    for processor in (SolarOpen2TemplateLogitProcessor().serialize(), "client-code"):
        req, params = _request()
        req.sampling_params = SamplingParams(custom_params=params, top_k=1)
        req.custom_logit_processor = processor
        req.return_sampling_mask = False
        req.sampling_logprobs_mode = "selected"
        req.origin_input_ids = [IDS.think_start]
        req.tokenizer = SimpleNamespace(eos_token_id=2, additional_stop_token_ids=None)
        reqs.append(req)
    batch = MagicMock()
    batch.reqs, batch.device = reqs, "cpu"
    # A client processor that slipped past TokenizerManager must not crash the scheduler.
    info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
    assert info.has_custom_logit_processor
    assert [p.rows for p in info.custom_logit_processor.values()] == [[0]]


def test_file_model_path_does_not_crash(tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")
    defaults = SolarOpen2ProcessorDefaults(
        model_path=str(model_file),
        tokenizer_path=str(model_file),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["SolarOpen2ForCausalLM"], eos_token_id=2
            ),
            hf_generation_config=None,
        ),
        tokenizer=None,
    )
    assert defaults.serialized is not None


def test_reasoner_grammar_accepts_forced_end_and_owns_content():
    from unittest.mock import MagicMock

    from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject

    inner = MagicMock()
    grammar = ReasonerGrammarObject(grammar=inner, think_end_ids=[IDS.think_end])
    grammar.maybe_init_reasoning(True)
    req, params = _request(structured=True, params={REASONING_BUDGET_EXTRA_ARG: 1})
    req.grammar = grammar
    req.output_ids.append(REGULAR_TOKEN)
    grammar.accept_token(REGULAR_TOKEN)
    inner.accept_token.assert_not_called()
    proc = SolarOpen2TemplateLogitProcessor()
    out = proc(_batch_logits(1), [params])
    assert torch.isfinite(out).sum() == 1
    req.output_ids.append(IDS.think_end)
    grammar.accept_token(IDS.think_end)
    assert grammar._is_generation()
    out = proc(_batch_logits(1), [params])
    assert torch.equal(out, _batch_logits(1))
    grammar.accept_token(REGULAR_TOKEN)
    inner.accept_token.assert_called_once_with(REGULAR_TOKEN)


def test_budget_out_of_vocab_and_index_cache():
    from sglang.srt.sampling.solar_open2 import _mask_index_tensor

    _, params = _request(output=[REGULAR_TOKEN], params={REASONING_BUDGET_EXTRA_ARG: 1})
    proc = SolarOpen2TemplateLogitProcessor()
    logits = torch.zeros(1, 131)  # think_end is outside this vocabulary
    proc(logits, [params])
    assert _is_masked(logits[0], IDS.im_end)
    assert _is_available(logits[0], REGULAR_TOKEN)
    first = _mask_index_tensor((2, 128, 999), 160, torch.device("cpu"))
    assert first is _mask_index_tensor((2, 128, 999), 160, torch.device("cpu"))
    assert first.tolist() == [2, 128]
    assert _mask_index_tensor((2, 128, 999), 100, torch.device("cpu")).tolist() == [2]


def test_server_cli_registers_solar_defaults():
    import argparse

    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "/unused"])
    assert args.solar_open2_reasoning_budget is None
    assert not args.disable_solar_open2_logits_processor
    args = parser.parse_args(
        [
            "--model-path",
            "/unused",
            "--solar-open2-reasoning-budget",
            "0",
            "--disable-solar-open2-logits-processor",
        ]
    )
    assert args.solar_open2_reasoning_budget == 0
    assert args.disable_solar_open2_logits_processor


class _RequestFixture:
    """Upstage's partial-call fixture translated to SGLang request parameters."""

    def __init__(
        self,
        token_ids,
        has_structured_outputs=False,
        extra_eos_token_ids=frozenset({2}),
        reasoning_budget=None,
        leading_forbidden_ids=frozenset(),
    ):
        self.token_ids = token_ids
        self.structured = has_structured_outputs
        self.eos = extra_eos_token_ids
        self.budget = reasoning_budget
        self.leading = leading_forbidden_ids

    def __call__(self, prompt, output, logits):
        return SolarOpen2TokenFSMEnforcer(
            prompt, self.token_ids, self.structured, self.eos, self.budget, self.leading
        )(prompt, output, logits)


class _UpstreamBatchFixture:
    """Only adapts upstream test inputs; masking runs in the real SGLang adapter."""

    def __init__(self, extra_eos):
        self.req_info = {}
        self.requests = {}
        self.extra_eos = extra_eos

    def apply(self, logits):
        params = []
        for row, partial in sorted(self.req_info.items()):
            factory = partial.func
            if row not in self.requests:
                self.requests[row] = _request(
                    *partial.args,
                    structured=factory.structured,
                    params={REASONING_BUDGET_EXTRA_ARG: factory.budget or 0},
                )
            params.append(self.requests[row][1])
        factory = next(iter(self.req_info.values())).func
        return SolarOpen2TemplateLogitProcessor(
            token_ids=factory.token_ids,
            extra_eos_token_ids=self.extra_eos,
            leading_forbidden_ids=factory.leading,
        )(logits, params)


def _make_processor(extra_eos=frozenset({EOS_ID})):
    return _UpstreamBatchFixture(extra_eos)


def _factory(has_structured_outputs=False):
    return _RequestFixture(IDS, has_structured_outputs=has_structured_outputs)


def _budget_factory(reasoning_budget):
    return _RequestFixture(IDS, reasoning_budget=reasoning_budget)


def test_batched_apply_matches_per_row_path():
    from functools import partial

    proc = _make_processor()
    outputs = []
    for req_idx, (prompt, output, structured) in enumerate(_BATCH_SCENARIOS):
        output = list(output)
        outputs.append(output)
        factory = _factory(structured)
        proc.req_info[req_idx] = partial(factory, prompt, output)

    batched = proc.apply(_batch_logits(len(_BATCH_SCENARIOS)))

    reference = _batch_logits(len(_BATCH_SCENARIOS))
    for req_idx, (prompt, output, structured) in enumerate(_BATCH_SCENARIOS):
        factory = _factory(structured)
        reference[req_idx] = factory(prompt, list(output), reference[req_idx])

    assert torch.equal(batched, reference)


def test_batched_apply_tracks_incremental_output_growth():
    from functools import partial

    proc = _make_processor()
    output: list[int] = []
    factory = _factory()
    proc.req_info[0] = partial(factory, [IDS.think_start], output)

    # Step 1: REASONING — only think_end among sentinels stays available.
    step1 = proc.apply(_batch_logits(1))
    assert _is_masked(step1[0], IDS.tool_call_start)
    assert _is_masked(step1[0], EOS_ID)
    assert _is_available(step1[0], IDS.think_end)

    # The adapter keeps a live reference to output ids; appending tokens
    # must advance the FSM on the next apply call.
    output.extend([IDS.think_end, REGULAR_TOKEN])
    step2 = proc.apply(_batch_logits(1))
    assert _is_available(step2[0], IDS.im_end)
    assert _is_available(step2[0], EOS_ID)
    assert _is_masked(step2[0], IDS.think_start)


def test_batched_apply_groups_rows_sharing_state():
    from functools import partial

    proc = _make_processor()
    # Two REASONING rows + one CONTENT-progress row -> two mask groups.
    scenarios = [
        ([IDS.think_start], []),
        ([IDS.think_start], []),
        ([IDS.think_start], [IDS.think_end, REGULAR_TOKEN]),
    ]
    for req_idx, (prompt, output) in enumerate(scenarios):
        proc.req_info[req_idx] = partial(_factory(), prompt, list(output))

    result = proc.apply(_batch_logits(3))
    for row in (0, 1):
        assert _is_masked(result[row], IDS.im_end)
        assert _is_masked(result[row], EOS_ID)
        assert _is_available(result[row], IDS.think_end)
    assert _is_available(result[2], IDS.im_end)
    assert _is_available(result[2], EOS_ID)
    assert _is_available(result[2], IDS.tool_call_start)


def test_batched_apply_forces_think_end_for_budget_exhausted_rows():
    from functools import partial

    proc = _make_processor()
    # Row 0: budget=2 exhausted (2 reasoning tokens) -> force <|think:end|>.
    # Row 1: no budget, still REASONING -> standard mask.
    proc.req_info[0] = partial(
        _budget_factory(2), [IDS.think_start], [REGULAR_TOKEN, REGULAR_TOKEN]
    )
    proc.req_info[1] = partial(
        _budget_factory(None), [IDS.think_start], [REGULAR_TOKEN]
    )

    result = proc.apply(_batch_logits(2))
    # Forced row: only think_end survives; regular text and sentinels masked.
    assert _is_available(result[0], IDS.think_end)
    assert _is_masked(result[0], REGULAR_TOKEN)
    assert _is_masked(result[0], IDS.im_end)
    assert _is_masked(result[0], EOS_ID)
    # Normal REASONING row: think_end and regular text stay available.
    assert _is_available(result[1], IDS.think_end)
    assert _is_available(result[1], REGULAR_TOKEN)
    assert _is_masked(result[1], IDS.im_end)


def test_batched_apply_budget_matches_per_row_path():
    from functools import partial

    scenarios: list[tuple[list[int], list[int], int | None]] = [
        ([IDS.think_start], [REGULAR_TOKEN, REGULAR_TOKEN], 2),  # forced
        ([IDS.think_start], [REGULAR_TOKEN], 2),  # below cap
        ([IDS.think_start], [REGULAR_TOKEN] * 4, None),  # unbounded
    ]
    proc = _make_processor()
    for req_idx, (prompt, output, budget) in enumerate(scenarios):
        proc.req_info[req_idx] = partial(_budget_factory(budget), prompt, list(output))
    batched = proc.apply(_batch_logits(len(scenarios)))

    reference = _batch_logits(len(scenarios))
    for req_idx, (prompt, output, budget) in enumerate(scenarios):
        reference[req_idx] = _budget_factory(budget)(
            prompt, list(output), reference[req_idx]
        )
    assert torch.equal(batched, reference)


def test_batched_apply_budget_force_falls_back_when_think_end_out_of_range():
    from functools import partial

    # Budget exhausted, but the vocab is too small to contain think_end (131):
    # the batched force path must fall back to normal REASONING masking, exactly
    # like the per-row __call__ does (no silently-unmasked row).
    proc = _make_processor()
    proc.req_info[0] = partial(_budget_factory(1), [IDS.think_start], [REGULAR_TOKEN])
    small_vocab = 130  # excludes think_end=131
    batched = proc.apply(
        torch.arange(small_vocab, dtype=torch.float32).reshape(1, small_vocab)
    )

    reference = torch.arange(small_vocab, dtype=torch.float32).reshape(1, small_vocab)
    reference[0] = _budget_factory(1)([IDS.think_start], [REGULAR_TOKEN], reference[0])
    assert torch.equal(batched, reference)
    # REASONING fallback masks im_end; regular text stays available.
    assert _is_masked(batched[0], IDS.im_end)
    assert _is_available(batched[0], REGULAR_TOKEN)


def test_batched_apply_masks_leading_newline():
    from functools import partial

    def leading_factory():
        return _RequestFixture(
            IDS,
            has_structured_outputs=False,
            extra_eos_token_ids=frozenset({EOS_ID}),
            leading_forbidden_ids=frozenset({NEWLINE_ID}),
        )

    proc = _make_processor()
    # Row 0: leading reasoning (prompt ends with think_start) -> newline masked.
    # Row 1: one reasoning token in -> newline available (not leading). The two
    # rows therefore land in distinct mask groups.
    proc.req_info[0] = partial(leading_factory(), [IDS.think_start], [])
    proc.req_info[1] = partial(leading_factory(), [IDS.think_start], [REGULAR_TOKEN])

    result = proc.apply(_batch_logits(2))
    assert _is_masked(result[0], NEWLINE_ID)
    assert _is_available(result[0], IDS.think_end)
    assert _is_available(result[1], NEWLINE_ID)

    # Batched masking must match the per-row __call__ path exactly.
    reference = _batch_logits(2)
    reference[0] = leading_factory()([IDS.think_start], [], reference[0])
    reference[1] = leading_factory()([IDS.think_start], [REGULAR_TOKEN], reference[1])
    assert torch.equal(result, reference)


def test_resolve_extra_eos_ids_from_generation_config(tmp_path):
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [2, 129]}')
    defaults = SolarOpen2ProcessorDefaults(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["SolarOpen2ForCausalLM"])
        ),
        model_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        tokenizer=None,
    )
    assert CustomLogitProcessor.from_str(
        defaults.serialized
    ).extra_eos_token_ids == frozenset({2})


def test_resolve_extra_eos_ids_env_override(tmp_path):
    from sglang.srt.environ import envs

    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [2]}')
    with envs.SGLANG_SOLAR_OPEN2_EOS_TOKEN_IDS.override("[5, 129]"):
        p = CustomLogitProcessor.from_str(_defaults(tmp_path).serialized)
        assert p.extra_eos_token_ids == frozenset({5})


def test_resolve_reasoning_budget_default_from_env():
    from sglang.srt.environ import envs
    from sglang.srt.sampling.solar_open2 import (
        DEFAULT_REASONING_BUDGET,
        resolve_solar_open2_reasoning_budget_default,
    )

    assert DEFAULT_REASONING_BUDGET == 128 * 1024
    for raw, expected in [
        (None, DEFAULT_REASONING_BUDGET),
        ("2048", 2048),
        ("0", None),
    ]:
        with envs.SGLANG_SOLAR_OPEN2_REASONING_BUDGET.override(raw):
            assert resolve_solar_open2_reasoning_budget_default() == expected
    with envs.SGLANG_SOLAR_OPEN2_REASONING_BUDGET.override("not-an-int"):
        with pytest.raises(ValueError):
            resolve_solar_open2_reasoning_budget_default()


def test_resolve_think_leading_forbidden_ids_from_vocab(tmp_path):
    from sglang.srt.environ import envs

    (tmp_path / "tokenizer.json").write_text(
        '{"model":{"vocab":{"Ċ":4294,"ĊĊ":4372,"ĊĊĊ":7183,"theĊ":5000,"good":6000}}}'
    )
    with envs.SGLANG_SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS.override(None):
        p = CustomLogitProcessor.from_str(_defaults(tmp_path).serialized)
        assert p.leading_forbidden_ids == frozenset({4294, 4372, 7183})


def test_resolve_think_leading_forbidden_ids_verbatim_newline_vocab(tmp_path):
    from sglang.srt.environ import envs

    (tmp_path / "tokenizer.json").write_text(
        '{"model":{"vocab":{"\\n":10,"\\n\\n":11,"x":12}}}'
    )
    with envs.SGLANG_SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS.override(None):
        p = CustomLogitProcessor.from_str(_defaults(tmp_path).serialized)
        assert p.leading_forbidden_ids == frozenset({10, 11})


def test_resolve_think_leading_forbidden_ids_default_fallback(tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.sampling.solar_open2 import _DEFAULT_LEADING_NEWLINE_IDS

    with envs.SGLANG_SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS.override(None):
        p = CustomLogitProcessor.from_str(_defaults(tmp_path).serialized)
        assert p.leading_forbidden_ids == _DEFAULT_LEADING_NEWLINE_IDS


def test_resolve_think_leading_forbidden_ids_env_override(tmp_path):
    from sglang.srt.environ import envs

    for raw, expected in [
        ("[4372, 4294]", frozenset({4372, 4294})),
        ("[]", frozenset()),
    ]:
        with envs.SGLANG_SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS.override(raw):
            p = CustomLogitProcessor.from_str(_defaults(tmp_path).serialized)
            assert p.leading_forbidden_ids == expected
    with envs.SGLANG_SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS.override("not-json"):
        with pytest.raises(ValueError):
            _defaults(tmp_path)


def test_metadata_override_precedence_and_validation(tmp_path):
    from sglang.srt.environ import envs

    (tmp_path / "tokenizer.json").write_text(
        '{"model":{"vocab":{"<|think:start|>":15}}}'
    )
    with envs.SGLANG_SOLAR_OPEN2_TOKEN_IDS.override(
        '{"think_start":10,"<|think:end|>":11}'
    ):
        p = CustomLogitProcessor.from_str(_defaults(tmp_path).serialized)
        assert p.token_ids.think_start == 10
        assert p.token_ids.think_end == 11
    for bad in ["not-json", "[]", '{"unknown":1}', '{"think_start":true}']:
        with envs.SGLANG_SOLAR_OPEN2_TOKEN_IDS.override(bad):
            with pytest.raises(ValueError):
                _defaults(tmp_path)
    with envs.SGLANG_SOLAR_OPEN2_REASONING_BUDGET.override("2048"):
        assert (
            CustomLogitProcessor.from_str(
                _defaults(tmp_path).serialized
            ).default_reasoning_budget
            == 2048
        )
        assert (
            CustomLogitProcessor.from_str(
                _defaults(tmp_path, budget=32).serialized
            ).default_reasoning_budget
            == 32
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
