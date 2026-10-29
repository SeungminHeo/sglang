"""Bitwise parity of xgrammar's batched mask fills with the serial per-row fills."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import xgrammar as xgr

from sglang.srt.constrained import xgrammar_backend
from sglang.srt.constrained.base_grammar_backend import GrammarRow
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import MAX_ROLLBACK_TOKENS, XGrammarGrammar
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    generate_chain_token_bitmask,
    generate_token_bitmask,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_VOCAB = [chr(c) for c in range(32, 127)] + ["</s>"]
_EOS = len(_VOCAB) - 1
# maxLength is what makes production mask fills slow, and it is a per-state constraint.
_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 8},
        "n": {"type": "integer"},
    },
    "required": ["name", "n"],
    "additionalProperties": False,
}


def _tokens(text):
    return [_VOCAB.index(c) for c in text]


class _XGrammarFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        info = xgr.TokenizerInfo(
            _VOCAB, vocab_type=xgr.VocabType.RAW, stop_token_ids=[_EOS]
        )
        cls.ctx = xgr.GrammarCompiler(info).compile_json_schema(
            json.dumps(_SCHEMA), any_whitespace=False
        )

    def _grammar(self, prefix=""):
        matcher = xgr.GrammarMatcher(self.ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        grammar = XGrammarGrammar(matcher, len(_VOCAB), self.ctx, None)
        for token in _tokens(prefix):
            grammar.accept_token(token)
        return grammar

    def _mask(self, rows):
        return xgrammar_backend._allocate_token_bitmask(len(_VOCAB), rows)


class TestXGrammarBatchedMask(_XGrammarFixture):
    PREFIXES = ["", "{", '{"name": "', '{"name": "abcdefgh', '{"name": "ab", "n": 1']

    def test_batched_matches_serial_on_sparse_rows(self):
        rows = [0, 2, 3, 5, 6]
        serial = self._mask(8)
        for row, prefix in zip(rows, self.PREFIXES):
            self._grammar(prefix).fill_vocab_mask(serial, row)

        batched = self._mask(8)
        entries = [
            GrammarRow(row=row, grammar=self._grammar(prefix))
            for row, prefix in zip(rows, self.PREFIXES)
        ]
        XGrammarGrammar.fill_vocab_mask_batched(entries, batched)

        self.assertTrue(torch.equal(serial, batched))
        self.assertTrue((batched[[1, 4, 7]] == -1).all())
        self.assertFalse(torch.equal(batched[3], batched[2]))

    def test_reasoner_rows_past_think_end_reach_batched_fill(self):
        """With a reasoning parser every grammar is a ReasonerGrammarObject; its
        generation-state rows must reach xgrammar's thread pool, not the per-row loop."""
        wrappers = []
        for thinking, prefix in ((False, "{"), (True, ""), (False, '{"name": "')):
            wrapper = ReasonerGrammarObject(self._grammar(prefix), think_end_ids=[0])
            wrapper.maybe_init_reasoning(thinking)
            wrappers.append(wrapper)

        serial = self._mask(3)
        for row, wrapper in enumerate(wrappers):
            wrapper.fill_vocab_mask(serial, row)

        spy = MagicMock(wraps=xgr.BatchGrammarMatcher(2))
        batched = self._mask(3)
        with patch.object(xgrammar_backend, "_batch_matcher", return_value=spy):
            ReasonerGrammarObject.fill_vocab_mask_batched(
                [GrammarRow(row=row, grammar=w) for row, w in enumerate(wrappers)],
                batched,
            )

        self.assertTrue(torch.equal(serial, batched))
        self.assertTrue((batched[1] == -1).all())
        self.assertEqual(spy.batch_fill_next_token_bitmask.call_args.args[2], [0, 2])


class TestChainTokenBitmask(_XGrammarFixture):
    """The batched chain fill must fill the same rows as the per-request DFS and
    leave every grammar where it started."""

    CASES = [
        # (committed prefix, chain: column 0 is the already-committed token)
        ("", '{"nam'),
        ('{"name": "abcdef', "fghij"),  # 'h' would be char 9 > maxLength: reject
        ('{"name": "ab", "n": ', "12}"),
        ('{"name": "ab", "n": 1', "}"),  # terminates, then a draft after the stop
    ]

    def _chain(self):
        rows = []
        for _, chain in self.CASES:
            ids = _tokens(chain)
            if chain == "}":
                ids = ids + [_EOS, _tokens("x")[0]]
            if chain == '{"nam':
                ids = ids + [-3]  # an out-of-range draft id is rejected, not indexed
            rows.append(ids + [_tokens(" ")[0]] * (6 - len(ids)))
        return torch.tensor(rows, dtype=torch.int64)

    def _reqs(self):
        reqs = []
        for prefix, chain in self.CASES:
            # Column 0 is already committed, so the grammar has accepted it too.
            reqs.append(SimpleNamespace(grammar=self._grammar(prefix + chain[0])))
        reqs.append(SimpleNamespace(grammar=None))
        return reqs

    def test_chain_fill_matches_dfs(self):
        verify_ids = torch.cat([self._chain(), self._chain()[:1]])
        serial_reqs, batched_reqs = self._reqs(), self._reqs()

        serial, _ = generate_token_bitmask(
            serial_reqs,
            *GrammarTree.from_linear_chain(verify_ids).resolve(),
            len(_VOCAB),
        )
        batched, _ = generate_chain_token_bitmask(
            reqs=batched_reqs, draft_tokens_cpu=verify_ids, vocab_size=len(_VOCAB)
        )

        self.assertTrue(torch.equal(serial, batched))
        for serial_req, batched_req in zip(serial_reqs[:-1], batched_reqs[:-1]):
            self.assertEqual(
                serial_req.grammar.accepted_tokens, batched_req.grammar.accepted_tokens
            )
            self.assertFalse(batched_req.grammar.is_terminated())


if __name__ == "__main__":
    unittest.main()
