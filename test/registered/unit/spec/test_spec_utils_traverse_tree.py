"""Regression test for spec_utils.traverse_tree calling xgrammar with tensors.

xgrammar 0.2.0 tightened its FFI binding and rejects 0-d tensors where Python
ints are expected. The dfs in traverse_tree recurses with `retrieve_next_token[curr]`
and reads `draft_tokens[curr]`, both of which return 0-d tensors and must be
explicitly cast before being handed to the grammar matcher.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.speculative.spec_utils import GrammarTree, traverse_tree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTraverseTreePassesIntsToGrammar(unittest.TestCase):
    def _record_grammar(self):
        """A grammar mock that records every call argument and rejects torch tensors."""
        grammar = MagicMock()
        grammar.is_terminated.return_value = False
        accept_calls = []
        fill_calls = []

        def record_accept(token):
            if isinstance(token, torch.Tensor):
                raise TypeError(f"accept_token got torch.Tensor: {token!r}")
            accept_calls.append(token)

        def record_fill(bitmask, idx):
            if isinstance(idx, torch.Tensor):
                raise TypeError(f"fill_vocab_mask got torch.Tensor idx: {idx!r}")
            fill_calls.append(idx)

        grammar.accept_token.side_effect = record_accept
        grammar.fill_vocab_mask.side_effect = record_fill
        grammar.rollback.return_value = None
        return grammar, accept_calls, fill_calls

    def _chain(self, verify_ids_2d):
        """Row 0 of the links chain-verify algorithms actually feed traverse_tree."""
        links = GrammarTree.from_linear_chain(verify_ids_2d).resolve()
        return tuple(t[0] for t in links)

    def test_branching_tree_passes_ints(self):
        # Binary tree exercises both child recursion and sibling recursion:
        #   0 ─┬─ 1
        #      └─ 2 ─── 3
        retrieve_next_token = torch.tensor([1, -1, 3, -1], dtype=torch.int32)
        retrieve_next_sibling = torch.tensor([-1, 2, -1, -1], dtype=torch.int32)
        draft_tokens = torch.tensor([100, 11, 22, 33], dtype=torch.int64)
        # all bits set: every draft token passes the parent's bitmask check
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar,
            bitmask,
        )

        self.assertEqual(set(accept_calls), {11, 22, 33})
        self.assertEqual(set(fill_calls), {0, 1, 2, 3})
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)

    def test_linear_chain_visits_all_positions_in_order(self):
        # Chain-verify algorithms (DFLASH/DSPARK) have no branching, so their tree
        # degenerates to 0 -- 1 -- 2 -- 3 with column 0 the already-committed token.
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 11, 22, 33]]))
        self.assertEqual(rnt.tolist(), [1, 2, 3, -1])
        self.assertEqual(rns.tolist(), [-1, -1, -1, -1])
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

        # Root (col 0) is never accepted; every draft token is, in chain order.
        self.assertEqual(accept_calls, [11, 22, 33])
        self.assertEqual(fill_calls, [0, 1, 2, 3])
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)

    def test_linear_chain_stops_at_grammar_reject(self):
        # A draft token the grammar disallows must stop the descent: no accept/fill
        # for that node or anything after it, so the mask rows past it stay unfilled
        # and only the already-filled prefix can be committed.
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 5, 7, 9]]))
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed
        # Disallow token id 7 (draft_tokens[2]) in node 1's mask (its parent).
        bitmask[1, 7 // 32] &= ~(1 << (7 % 32))

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

        # Node 1 accepted+filled; node 2 rejected -> node 2 and node 3 skipped.
        self.assertEqual(accept_calls, [5])
        self.assertEqual(fill_calls, [0, 1])


class TestTraverseTreeRejectsOutOfRangeDraftTokens(unittest.TestCase):
    """A draft token outside [0, vocab_size) must be rejected, not indexed with.

    Production hit this with structured output on: the fused top-k=1 draft
    returned an id read past the end of its `partial_indices` buffer, so
    `draft_tokens` carried values like -1088176107. The upper-bound check let
    every negative through, and `parent_bitmask[current_token // 32]` then
    raised IndexError inside this DFS, taking down all TP ranks.
    """

    VOCAB = 163840  # A.X-K2

    def _grammar(self):
        grammar = MagicMock()
        grammar.is_terminated.return_value = False
        accept_calls, fill_calls = [], []
        grammar.accept_token.side_effect = accept_calls.append
        grammar.fill_vocab_mask.side_effect = lambda _b, idx: fill_calls.append(idx)
        grammar.rollback.return_value = None
        return grammar, accept_calls, fill_calls

    def _run(self, bad_token):
        # chain 0 -- 1 -- 2 -- 3, with the bad id at node 2
        rnt = torch.tensor([1, 2, 3, -1], dtype=torch.int32)
        rns = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
        draft_tokens = torch.tensor([100, 11, bad_token, 33], dtype=torch.int64)
        bitmask = torch.full((4, self.VOCAB // 32), -1, dtype=torch.int32)
        grammar, accept_calls, fill_calls = self._grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask, vocab_size=self.VOCAB)
        return accept_calls, fill_calls

    def test_large_negative_token_is_rejected(self):
        accept_calls, fill_calls = self._run(-1088176107)
        # node 2 rejected -> it and node 3 are skipped, prefix stays usable
        self.assertEqual(accept_calls, [11])
        self.assertEqual(fill_calls, [0, 1])

    def test_minus_one_sentinel_is_rejected(self):
        # -1 is the sentinel these tree tensors use elsewhere; as a token id it
        # would silently index the last bitmask word instead of raising.
        accept_calls, fill_calls = self._run(-1)
        self.assertEqual(accept_calls, [11])
        self.assertEqual(fill_calls, [0, 1])

    def test_token_at_or_above_vocab_size_is_rejected(self):
        accept_calls, fill_calls = self._run(self.VOCAB)
        self.assertEqual(accept_calls, [11])
        self.assertEqual(fill_calls, [0, 1])

    def test_in_range_token_still_accepted(self):
        accept_calls, fill_calls = self._run(self.VOCAB - 1)
        self.assertEqual(accept_calls, [11, self.VOCAB - 1, 33])
        self.assertEqual(fill_calls, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
