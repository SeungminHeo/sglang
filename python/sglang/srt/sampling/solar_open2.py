# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Solar Open2 template enforcement.

FSM and grouped masking ported from UpstageAI/vllm v0.22.0-solar-open2,
vllm/v1/sample/logits_processor/solar_open2.py. Request state lives on Req,
not on the processor instance (which may be reconstructed for each batch).
"""

import json
import logging
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

logger = logging.getLogger(__name__)
NEG_INF = float("-inf")
DEFAULT_REASONING_BUDGET = 128 * 1024
DISABLE_EXTRA_ARG = "disable_solar_open2_logits_processor"
REASONING_BUDGET_EXTRA_ARG = "solar_open2_reasoning_budget"
_DEFAULT_LEADING_NEWLINE_IDS = frozenset({4294, 4372})
_FORCE_THINK_END = "force_think_end"
# This prefix describes trusted built-in code, never a pickled callable.
SOLAR_PROCESSOR_PREFIX = "sglang:solar-open2:v1:"


@dataclass(frozen=True)
class SolarOpen2TokenIds:
    """Single-token control IDs for the Solar Open2 chat envelope.

    Defaults match the Solar Open2 tokenizer layout. Runtime tokenizer metadata overrides them.
    """

    im_start: int | None = 128
    im_end: int | None = 129
    think_start: int | None = 130
    think_end: int | None = 131
    im_content: int | None = 132
    tool_start: int | None = 133
    tool_end: int | None = 134
    tool_call_start: int | None = 135
    tool_call_end: int | None = 136
    tool_arg_start: int | None = 137
    tool_arg_value: int | None = 138
    tool_arg_end: int | None = 139
    tool_response_start: int | None = 140
    tool_response_end: int | None = 141


_TOKEN_TEXT_BY_FIELD = {
    "im_start": "<|im:start|>",
    "im_end": "<|im:end|>",
    "think_start": "<|think:start|>",
    "think_end": "<|think:end|>",
    "im_content": "<|im:content|>",
    "tool_start": "<|tool:start|>",
    "tool_end": "<|tool:end|>",
    "tool_call_start": "<|tool_call:start|>",
    "tool_call_end": "<|tool_call:end|>",
    "tool_arg_start": "<|tool_arg:start|>",
    "tool_arg_value": "<|tool_arg:value|>",
    "tool_arg_end": "<|tool_arg:end|>",
    "tool_response_start": "<|tool_response:start|>",
    "tool_response_end": "<|tool_response:end|>",
}


class SolarOpen2State(Enum):
    REASONING = "reasoning"
    CONTENT = "content"
    TOOL_CALL_BEGIN = "tool_call_begin"
    TOOL_CALL_NAME = "tool_call_name"
    TOOL_ARG_BEGIN = "tool_arg_begin"
    TOOL_ARG_NAME = "tool_arg_name"
    TOOL_ARG_VALUE_BEGIN = "tool_arg_value_begin"
    TOOL_ARG_VALUE = "tool_arg_value"
    TOOL_ARG_END = "tool_arg_end"
    TOOL_CALL_END = "tool_call_end"


def _read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, NotADirectoryError):
        # A file --model-path makes "<file>/generation_config.json" a non-directory path.
        return None
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse %s for Solar Open2 token IDs: %s", path, e)
        return None


def _token_ids_from_json(data: Any) -> dict[str, int]:
    """Extract token text -> id mappings from common HF tokenizer JSON shapes."""
    token_to_id: dict[str, int] = {}
    if not isinstance(data, dict):
        return token_to_id

    added_decoder = data.get("added_tokens_decoder")
    if isinstance(added_decoder, dict):
        for raw_id, raw_token in added_decoder.items():
            if isinstance(raw_token, dict):
                content = raw_token.get("content")
            else:
                content = raw_token
            if isinstance(content, str):
                token_to_id[content] = int(raw_id)

    added_tokens = data.get("added_tokens")
    if isinstance(added_tokens, list):
        for raw_token in added_tokens:
            if not isinstance(raw_token, dict):
                continue
            content = raw_token.get("content")
            raw_id = raw_token.get("id")
            if isinstance(content, str) and raw_id is not None:
                token_to_id[content] = int(raw_id)

    model = data.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        for token, raw_id in vocab.items():
            if isinstance(token, str) and raw_id is not None:
                token_to_id[token] = int(raw_id)

    return token_to_id


def _resolve_token_ids_from_tokenizer_dirs(
    tokenizer_dirs: list[Path],
) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for tokenizer_dir in tokenizer_dirs:
        token_to_id: dict[str, int] = {}
        for name in (
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
        ):
            token_to_id.update(_token_ids_from_json(_read_json(tokenizer_dir / name)))
        for field_name, token_text in _TOKEN_TEXT_BY_FIELD.items():
            if field_name not in resolved and token_text in token_to_id:
                resolved[field_name] = token_to_id[token_text]
    return resolved


def _coerce_eos_ids(value: Any) -> set[int]:
    """Normalize an ``eos_token_id`` value (int | list | None) to a set."""
    if value is None or isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        return {int(v) for v in value if isinstance(v, int) and not isinstance(v, bool)}
    return set()


def _sentinel_id_set(token_ids: SolarOpen2TokenIds) -> set[int]:
    return {
        getattr(token_ids, field.name)
        for field in fields(SolarOpen2TokenIds)
        if getattr(token_ids, field.name) is not None
    }


_MASK_SPEC_BY_STATE: dict[SolarOpen2State, tuple[tuple[str, ...], bool]] = {
    SolarOpen2State.REASONING: (("think_end",), True),
    SolarOpen2State.TOOL_CALL_BEGIN: ((), True),
    SolarOpen2State.TOOL_CALL_NAME: (("tool_arg_start", "tool_call_end"), True),
    SolarOpen2State.TOOL_ARG_BEGIN: ((), True),
    SolarOpen2State.TOOL_ARG_NAME: (("tool_arg_value",), True),
    SolarOpen2State.TOOL_ARG_VALUE_BEGIN: (("tool_arg_end",), True),
    SolarOpen2State.TOOL_ARG_VALUE: (("tool_arg_end",), True),
    SolarOpen2State.TOOL_ARG_END: (("tool_arg_start", "tool_call_end"), True),
    SolarOpen2State.TOOL_CALL_END: (("tool_call_start", "im_end"), False),
}


_MASK_SPEC_CONTENT: dict[bool, tuple[tuple[str, ...], bool]] = {
    # content_progress=True: the turn may legally end -> EOS stays available.
    True: (("tool_call_start", "im_end"), False),
    # Fresh CONTENT (no content yet): turn end (im_end + bare EOS) forbidden.
    False: (("tool_call_start",), True),
}


@lru_cache(maxsize=8)
def _forbidden_table(
    token_ids: SolarOpen2TokenIds,
    extra_eos_token_ids: frozenset[int],
) -> dict[tuple[SolarOpen2State, bool], tuple[int, ...]]:
    """Precompute, per (state, content_progress), the forbidden token ids.

    The forbidden set for a decoding step is fully determined by the FSM
    state (plus the content-progress flag in CONTENT), so it can be built
    once per (token_ids, extra_eos) configuration instead of re-deriving
    Python lists on every step of every request.
    """
    all_controls = {
        getattr(token_ids, field.name)
        for field in fields(SolarOpen2TokenIds)
        if getattr(token_ids, field.name) is not None
    }

    def build(allowed_fields: tuple[str, ...], eos_masked: bool) -> tuple[int, ...]:
        allowed = {
            token_id
            for token_id in (getattr(token_ids, name) for name in allowed_fields)
            if token_id is not None
        }
        forbidden = all_controls - allowed
        if eos_masked:
            forbidden |= extra_eos_token_ids
        return tuple(sorted(forbidden))

    table: dict[tuple[SolarOpen2State, bool], tuple[int, ...]] = {}
    for state, (allowed_fields, eos_masked) in _MASK_SPEC_BY_STATE.items():
        ids = build(allowed_fields, eos_masked)
        table[(state, False)] = ids
        table[(state, True)] = ids
    for progress, (allowed_fields, eos_masked) in _MASK_SPEC_CONTENT.items():
        table[(SolarOpen2State.CONTENT, progress)] = build(allowed_fields, eos_masked)
    return table


_MASK_TENSOR_CACHE: dict[tuple[tuple[int, ...], int, torch.device], torch.Tensor] = {}


def _mask_index_tensor(
    forbidden_ids: tuple[int, ...],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    key = (forbidden_ids, vocab_size, device)
    tensor = _MASK_TENSOR_CACHE.get(key)
    if tensor is None:
        in_vocab = [
            token_id for token_id in forbidden_ids if 0 <= token_id < vocab_size
        ]
        tensor = torch.tensor(in_vocab, dtype=torch.long, device=device)
        _MASK_TENSOR_CACHE[key] = tensor
    return tensor


def _normalize_reasoning_budget(value: Any) -> int | None:
    """Coerce a budget value to a positive int, or ``None`` (= no cap).

    Accepts ``int`` (``bool`` rejected — a budget is a token count, not a
    toggle). ``<= 0`` collapses to ``None`` so callers can disable the cap with
    ``0`` (e.g. a per-request override of a server-wide default).
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{REASONING_BUDGET_EXTRA_ARG} must be a non-negative integer "
            "(reasoning-token count; 0 or omitted disables the cap)"
        )
    if value < 0:
        raise ValueError(
            f"{REASONING_BUDGET_EXTRA_ARG} must be non-negative, got {value}"
        )
    return value if value > 0 else None


class SolarOpen2TokenFSMEnforcer:
    """Request-level Solar Open2 token finite-state-machine enforcer.

    Masks structurally invalid single-token sentinels for the current phase.
    Optionally enforces a reasoning-token budget: once ``reasoning_budget``
    tokens have been emitted inside a ``<|think:start|>`` / ``<|think:end|>``
    block, the FSM force-emits ``<|think:end|>`` to end reasoning. ``None`` (the
    default) disables the budget — pure structural enforcement.
    """

    def __init__(
        self,
        prompt_token_ids: list[int],
        token_ids: SolarOpen2TokenIds,
        has_structured_outputs: bool = False,
        extra_eos_token_ids: frozenset[int] = frozenset(),
        reasoning_budget: int | None = None,
        leading_forbidden_ids: frozenset[int] = frozenset(),
    ) -> None:
        self.token_ids = token_ids
        self._initial_prompt_state = self._initial_state(prompt_token_ids)
        self._state = self._initial_prompt_state
        self._last_processed_len = 0
        # Whether the token immediately preceding the next decode step is
        # <|think:start|> (i.e. the upcoming token is the *leading* reasoning
        # token). Seeded from the prompt's last token so a template-appended
        # <|think:start|> is caught on the very first generated token, then kept
        # in sync as output tokens are processed.
        think_start = token_ids.think_start
        self._initial_prev_is_think_start = (
            think_start is not None
            and bool(prompt_token_ids)
            and prompt_token_ids[-1] == think_start
        )
        self._prev_was_think_start = self._initial_prev_is_think_start
        self._has_structured_outputs = has_structured_outputs
        # Max reasoning tokens before the FSM force-emits <|think:end|>. None or
        # <= 0 disables the cap. Only counts tokens generated in the current
        # turn while in REASONING; the think_start / think_end sentinels are
        # excluded, and reasoning already present in the prompt is not counted
        # (the cap bounds *new* reasoning this generation adds).
        self._reasoning_budget = _normalize_reasoning_budget(reasoning_budget)
        self._reasoning_token_count = 0
        # Non-sentinel EOS ids (e.g. <|endoftext|>): allowed only in states
        # where ending the turn is template-legal (see __call__). Without
        # this, the model can stop mid-reasoning / mid-tool-call, which the
        # non-streaming parser papers over but the streaming parser cannot.
        self._extra_eos_token_ids = extra_eos_token_ids
        # True once the turn has produced actual content: at least one
        # regular token while in CONTENT, or a completed tool call. Until
        # then, ending the turn from CONTENT is forbidden — prevents
        # ``<|think:end|><|im:end|>`` (empty assistant message).
        self._content_progress = False
        self._all_control_token_ids = frozenset(
            token_id
            for token_id in (
                token_ids.im_start,
                token_ids.im_end,
                token_ids.think_start,
                token_ids.think_end,
                token_ids.im_content,
                token_ids.tool_start,
                token_ids.tool_end,
                token_ids.tool_call_start,
                token_ids.tool_call_end,
                token_ids.tool_arg_start,
                token_ids.tool_arg_value,
                token_ids.tool_arg_end,
                token_ids.tool_response_start,
                token_ids.tool_response_end,
            )
            if token_id is not None
        )
        # Forbidden-id table shared per (token_ids, eos) configuration.
        self._forbidden_by_key = _forbidden_table(token_ids, extra_eos_token_ids)
        # Forbidden set for the *leading* reasoning token (right after
        # <|think:start|>): the normal REASONING mask plus the leak-triggering
        # newline ids. None when no leading ids are configured, so the leading
        # step is indistinguishable from any other REASONING step.
        base_reasoning = self._forbidden_by_key[(SolarOpen2State.REASONING, False)]
        self._reasoning_leading_forbidden: tuple[int, ...] | None = (
            tuple(sorted(set(base_reasoning) | set(leading_forbidden_ids)))
            if leading_forbidden_ids
            else None
        )
        # token id -> next state for state-changing sentinels. ``setdefault``
        # preserves the original elif-chain precedence if two fields were
        # misconfigured with the same id.
        transitions: dict[int, SolarOpen2State] = {}
        for token_id, next_state in (
            (token_ids.think_start, SolarOpen2State.REASONING),
            (token_ids.think_end, SolarOpen2State.CONTENT),
            (token_ids.tool_call_start, SolarOpen2State.TOOL_CALL_BEGIN),
            (token_ids.tool_call_end, SolarOpen2State.TOOL_CALL_END),
            (token_ids.tool_arg_start, SolarOpen2State.TOOL_ARG_BEGIN),
            (token_ids.tool_arg_value, SolarOpen2State.TOOL_ARG_VALUE_BEGIN),
            (token_ids.tool_arg_end, SolarOpen2State.TOOL_ARG_END),
        ):
            if token_id is not None:
                transitions.setdefault(token_id, next_state)
        self._sentinel_transitions = transitions

    def _initial_state(self, prompt_token_ids: list[int]) -> SolarOpen2State:
        think_start = self.token_ids.think_start
        think_end = self.token_ids.think_end
        if think_start is None or think_end is None:
            return SolarOpen2State.CONTENT

        last_start = _rindex(prompt_token_ids, think_start)
        if last_start is None:
            return SolarOpen2State.CONTENT
        last_end = _rindex(prompt_token_ids, think_end)
        if last_end is None or last_start > last_end:
            return SolarOpen2State.REASONING
        return SolarOpen2State.CONTENT

    def _reset_to_prompt_state(self) -> None:
        self._state = self._initial_prompt_state
        self._content_progress = False
        self._reasoning_token_count = 0
        self._last_processed_len = 0
        self._prev_was_think_start = self._initial_prev_is_think_start

    def _process_token(self, token_id: int) -> None:
        ids = self.token_ids
        prev_state = self._state
        next_state = self._sentinel_transitions.get(token_id)
        if next_state is not None:
            self._state = next_state
        elif self._state == SolarOpen2State.TOOL_CALL_BEGIN:
            self._state = SolarOpen2State.TOOL_CALL_NAME
        elif self._state == SolarOpen2State.TOOL_ARG_BEGIN:
            self._state = SolarOpen2State.TOOL_ARG_NAME
        elif self._state == SolarOpen2State.TOOL_ARG_VALUE_BEGIN:
            self._state = SolarOpen2State.TOOL_ARG_VALUE
        elif self._state == SolarOpen2State.TOOL_CALL_END:
            self._state = SolarOpen2State.CONTENT

        # Reasoning-budget accounting: count tokens emitted *inside* a reasoning
        # block this turn. <|think:start|> (re)opens the block and resets the
        # counter; only tokens that keep the FSM in REASONING count (the closing
        # <|think:end|> transitions to CONTENT and is excluded).
        if ids.think_start is not None and token_id == ids.think_start:
            self._reasoning_token_count = 0
        elif (
            prev_state == SolarOpen2State.REASONING
            and self._state == SolarOpen2State.REASONING
        ):
            self._reasoning_token_count += 1

        if ids.tool_call_end is not None and token_id == ids.tool_call_end:
            # A completed tool call counts as turn content.
            self._content_progress = True
        elif (
            self._state == SolarOpen2State.CONTENT
            and token_id not in self._all_control_token_ids
        ):
            self._content_progress = True

        # Track whether *this* token is <|think:start|> so the next decode step
        # can recognize the leading reasoning position (the token right after it).
        self._prev_was_think_start = (
            ids.think_start is not None and token_id == ids.think_start
        )

    def _update_state_incremental(self, output_token_ids: list[int]) -> None:
        current_len = len(output_token_ids)
        if current_len < self._last_processed_len:
            self._reset_to_prompt_state()
        for idx in range(self._last_processed_len, current_len):
            self._process_token(output_token_ids[idx])
        self._last_processed_len = current_len

    def _reasoning_budget_exhausted(self) -> bool:
        """True when the reasoning-token cap has been reached this turn."""
        return (
            self._reasoning_budget is not None
            and self._reasoning_token_count >= self._reasoning_budget
        )

    @staticmethod
    def _force_single_token(logits: torch.Tensor, token_id: int) -> None:
        """Mask the entire vocabulary except ``token_id`` (force its emission)."""
        keep = logits[token_id].item()
        logits.fill_(NEG_INF)
        logits[token_id] = keep

    def advance_mask_ids(
        self, output_token_ids: list[int]
    ) -> tuple[int, ...] | str | None:
        """Advance the FSM and return this step's masking directive.

        Returns one of:
        - a forbidden-id tuple: the precomputed union of (a) control sentinels
          structurally invalid for the current state and (b) bare EOS ids
          (e.g. ``<|endoftext|>``) wherever ending the turn is template-illegal
          — anywhere except CONTENT-with-progress and right after a closed tool
          call — forcing the model to emit ``<|think:end|>`` (and close any open
          tool call) before it can stop, so streaming and non-streaming parses
          stay equivalent;
        - ``_FORCE_THINK_END``: the reasoning budget is exhausted, so this step
          must force ``<|think:end|>`` (mask everything except think_end);
        - ``None``: no masking applies (structured outputs own the CONTENT
          phase).
        """
        self._update_state_incremental(output_token_ids)
        state = self._state
        if self._has_structured_outputs and state == SolarOpen2State.CONTENT:
            return None
        if state == SolarOpen2State.REASONING:
            if (
                self._reasoning_budget_exhausted()
                and self.token_ids.think_end is not None
            ):
                return _FORCE_THINK_END
            # Leading reasoning token (the one right after <|think:start|>):
            # forbid the leak-triggering newline ids on top of the normal
            # REASONING sentinel/EOS mask. Deeper reasoning tokens are untouched.
            if (
                self._prev_was_think_start
                and self._reasoning_leading_forbidden is not None
            ):
                return self._reasoning_leading_forbidden
            return self._forbidden_by_key[(SolarOpen2State.REASONING, False)]
        progress = state == SolarOpen2State.CONTENT and self._content_progress
        return self._forbidden_by_key[(state, progress)]

    def __call__(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        del prompt_token_ids
        directive = self.advance_mask_ids(output_token_ids)
        if directive is None:
            return logits
        if isinstance(directive, str):
            # Reasoning-budget cap: force <|think:end|> so the model stops
            # reasoning. If think_end is unresolved/out of range the cap is
            # unenforceable, so fall back to normal reasoning masking.
            think_end = self.token_ids.think_end
            if think_end is not None and think_end < logits.shape[-1]:
                self._force_single_token(logits, think_end)
                return logits
            forbidden_ids = self._forbidden_by_key[(SolarOpen2State.REASONING, False)]
        else:
            forbidden_ids = directive
        if forbidden_ids:
            indices = _mask_index_tensor(forbidden_ids, logits.shape[-1], logits.device)
            if indices.numel():
                logits[indices] = NEG_INF
        return logits


def _rindex(values: list[int], needle: int) -> int | None:
    for idx in range(len(values) - 1, -1, -1):
        if values[idx] == needle:
            return idx
    return None


class SolarOpen2TemplateLogitProcessor(CustomLogitProcessor):
    """Batch adapter; only Req owns mutable decoding state."""

    supports_speculative = False

    def __init__(
        self,
        token_ids: SolarOpen2TokenIds = SolarOpen2TokenIds(),
        extra_eos_token_ids: frozenset[int] = frozenset({2}),
        default_reasoning_budget: int | None = DEFAULT_REASONING_BUDGET,
        leading_forbidden_ids: frozenset[int] = _DEFAULT_LEADING_NEWLINE_IDS,
    ):
        self.token_ids = token_ids
        self.extra_eos_token_ids = extra_eos_token_ids
        self.default_reasoning_budget = _normalize_reasoning_budget(
            default_reasoning_budget
        )
        self.leading_forbidden_ids = leading_forbidden_ids

    @staticmethod
    def validate_params(params):
        if params is not None and not isinstance(params, dict):
            raise ValueError("Solar Open2 custom_params must be an object")
        params = params or {}
        value = params.get(DISABLE_EXTRA_ARG)
        if value is not None and not isinstance(value, int):
            raise ValueError(f"{DISABLE_EXTRA_ARG} must be an integer")
        _normalize_reasoning_budget(params.get(REASONING_BUDGET_EXTRA_ARG))

    def serialize(self) -> str:
        return SOLAR_PROCESSOR_PREFIX + json.dumps(
            {
                "token_ids": asdict(self.token_ids),
                "extra_eos_token_ids": sorted(self.extra_eos_token_ids),
                "default_reasoning_budget": self.default_reasoning_budget,
                "leading_forbidden_ids": sorted(self.leading_forbidden_ids),
            },
            sort_keys=True,
        )

    @classmethod
    def deserialize(cls, value: str):
        config = json.loads(value[len(SOLAR_PROCESSOR_PREFIX) :])
        config["token_ids"] = SolarOpen2TokenIds(**config["token_ids"])
        for name in ("extra_eos_token_ids", "leading_forbidden_ids"):
            config[name] = frozenset(config[name])
        return cls(**config)

    def _enforcer(self, params):
        req = params["__req__"]
        enforcer = getattr(req, "_solar_open2_enforcer", None)
        if enforcer is None:
            budget = params.get(REASONING_BUDGET_EXTRA_ARG)
            if budget is None:
                budget = self.default_reasoning_budget
            sampling = req.sampling_params
            # A ReasonerGrammarObject may exist solely for reasoning control.
            # Only actual output constraints own the CONTENT phase.
            structured = any(
                getattr(sampling, name, None) is not None
                for name in ("json_schema", "regex", "ebnf", "structural_tag")
            )
            enforcer = SolarOpen2TokenFSMEnforcer(
                req.origin_input_ids,
                self.token_ids,
                has_structured_outputs=structured,
                extra_eos_token_ids=self.extra_eos_token_ids,
                reasoning_budget=budget,
                leading_forbidden_ids=self.leading_forbidden_ids,
            )
            req._solar_open2_enforcer = enforcer
        return enforcer

    def __call__(self, logits, custom_param_list=None):
        if not custom_param_list:
            return logits
        if len(custom_param_list) != logits.shape[0]:
            raise ValueError("Solar Open2 requires one logit row per request")
        rows_by_ids = {}
        force_rows = []
        seen = set()
        for row, params in enumerate(custom_param_list):
            if params.get(DISABLE_EXTRA_ARG):
                continue
            req = params["__req__"]
            # Speculative expansion repeats the same request on multiple rows.
            # It needs tentative branch histories, which this adapter cannot infer.
            identity = id(params)
            if identity in seen:
                raise ValueError("Solar Open2 does not support speculative decoding")
            seen.add(identity)
            directive = self._enforcer(params).advance_mask_ids(req.output_ids)
            if directive is None:
                continue
            if directive == _FORCE_THINK_END:
                force_rows.append(row)
            else:
                rows_by_ids.setdefault(directive, []).append(row)
        vocab_size, device = logits.shape[-1], logits.device
        for forbidden_ids, rows in rows_by_ids.items():
            cols = _mask_index_tensor(forbidden_ids, vocab_size, device)
            if cols.numel():
                rows_t = torch.tensor(rows, dtype=torch.long, device=device)
                logits[rows_t.unsqueeze(1), cols.unsqueeze(0)] = NEG_INF
        if force_rows:
            rows_t = torch.tensor(force_rows, dtype=torch.long, device=device)
            think_end = self.token_ids.think_end
            if think_end is not None and 0 <= think_end < vocab_size:
                saved = logits[rows_t, think_end].clone()
                logits[rows_t, :] = NEG_INF
                logits[rows_t, think_end] = saved
            else:
                forbidden = _forbidden_table(self.token_ids, self.extra_eos_token_ids)[
                    (SolarOpen2State.REASONING, False)
                ]
                cols = _mask_index_tensor(forbidden, vocab_size, device)
                if cols.numel():
                    logits[rows_t.unsqueeze(1), cols.unsqueeze(0)] = NEG_INF
        return logits


def _int_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid token id")
    return int(value)


def _resolve_token_ids_from_env():
    raw = envs.SGLANG_SOLAR_OPEN2_TOKEN_IDS.get()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError("SGLANG_SOLAR_OPEN2_TOKEN_IDS must be a JSON object") from e
    if not isinstance(data, dict):
        raise ValueError("SGLANG_SOLAR_OPEN2_TOKEN_IDS must be a JSON object")
    by_text = {text: field for field, text in _TOKEN_TEXT_BY_FIELD.items()}
    resolved = {}
    for key, value in data.items():
        field = by_text.get(key, key)
        if field not in _TOKEN_TEXT_BY_FIELD:
            raise ValueError(f"Unknown Solar Open2 token id field: {key}")
        resolved[field] = _int_or_none(value)
    return resolved


def _resolve_id_list_override(field):
    raw = field.get()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{field.name} must be a JSON list of ints") from e
    if not isinstance(data, list):
        raise ValueError(f"{field.name} must be a JSON list of ints")
    return {value for value in (_int_or_none(v) for v in data) if value is not None}


def resolve_solar_open2_reasoning_budget_default():
    raw = envs.SGLANG_SOLAR_OPEN2_REASONING_BUDGET.get()
    if not raw:
        return DEFAULT_REASONING_BUDGET
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            "SGLANG_SOLAR_OPEN2_REASONING_BUDGET must be an integer number of tokens"
        ) from e
    return _normalize_reasoning_budget(value)


def resolve_solar_open2_think_leading_forbidden_ids(token_to_id):
    override = _resolve_id_list_override(
        envs.SGLANG_SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS
    )
    if override is not None:
        return frozenset(override)
    ids = {idx for text, idx in token_to_id.items() if text and set(text) == {"Ċ"}}
    for literal in ("\n", "\n\n"):
        if literal in token_to_id:
            ids.add(token_to_id[literal])
    return frozenset(ids) or _DEFAULT_LEADING_NEWLINE_IDS


class SolarOpen2ProcessorDefaults:
    """Resolve model metadata once and attach the built-in at the trust boundary."""

    def __init__(
        self,
        *,
        model_config,
        tokenizer,
        model_path,
        tokenizer_path,
        disabled=False,
        reasoning_budget=None,
        speculative_algorithm=None,
    ):
        self.serialized = None
        architectures = getattr(model_config.hf_config, "architectures", ()) or ()
        if "SolarOpen2ForCausalLM" not in architectures or disabled:
            return
        if speculative_algorithm:
            raise ValueError(
                "Solar Open2 template logits processing does not support speculative "
                "decoding; use --disable-solar-open2-logits-processor to opt out"
            )
        budget = (
            resolve_solar_open2_reasoning_budget_default()
            if reasoning_budget is None
            else _normalize_reasoning_budget(reasoning_budget)
        )
        token_to_id = {}
        # Local metadata also works with --skip-tokenizer-init. A loaded tokenizer
        # supplies remote HF metadata and takes precedence over local fallbacks.
        for value in (model_path, tokenizer_path):
            if value:
                directory = Path(value)
                if directory.is_dir():
                    for name in (
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                    ):
                        token_to_id.update(
                            _token_ids_from_json(_read_json(directory / name))
                        )
        if tokenizer is not None:
            token_to_id.update(tokenizer.get_vocab())
        resolved = {
            field: token_to_id[text]
            for field, text in _TOKEN_TEXT_BY_FIELD.items()
            if text in token_to_id
        }
        resolved.update(_resolve_token_ids_from_env())
        token_ids = SolarOpen2TokenIds(**resolved)
        eos = set()
        for config in (
            model_config.hf_config,
            getattr(model_config, "hf_generation_config", None),
        ):
            eos.update(_coerce_eos_ids(getattr(config, "eos_token_id", None)))
        for value in (model_path, tokenizer_path):
            if value:
                generation = _read_json(Path(value) / "generation_config.json")
                if isinstance(generation, dict):
                    eos.update(_coerce_eos_ids(generation.get("eos_token_id")))
        eos.update(_coerce_eos_ids(getattr(tokenizer, "eos_token_id", None)))
        if "<|endoftext|>" in token_to_id:
            eos.add(token_to_id["<|endoftext|>"])
        if not eos:
            eos.add(2)
        eos_override = _resolve_id_list_override(envs.SGLANG_SOLAR_OPEN2_EOS_TOKEN_IDS)
        if eos_override is not None:
            eos = eos_override
        newline_ids = resolve_solar_open2_think_leading_forbidden_ids(token_to_id)
        processor = SolarOpen2TemplateLogitProcessor(
            token_ids, frozenset(eos - _sentinel_id_set(token_ids)), budget, newline_ids
        )
        self.serialized = processor.serialize()
        logger.info(
            "Solar Open2 template logits processor enabled (reasoning budget=%s)",
            budget,
        )

    def for_request(self, client_processor, sampling_params):
        if self.serialized is None:
            return client_processor
        params = sampling_params.custom_params
        SolarOpen2TemplateLogitProcessor.validate_params(params)
        if params and params.get(DISABLE_EXTRA_ARG):
            return client_processor
        if client_processor is not None:
            raise ValueError(
                "A client custom_logit_processor cannot be combined with the default "
                "Solar Open2 processor; set disable_solar_open2_logits_processor in "
                "custom_params to use the client processor"
            )
        # Req installs its weak reference only when custom_params is a dict.
        sampling_params.custom_params = dict(params or {})
        return self.serialized
