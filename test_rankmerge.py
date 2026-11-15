"""Verification suite for rankmerge (standard-library unittest).

Covers: the Source/handle contract, an independent full-materialisation
reference (differential test), ties, updates interleaved with reads, empty
sources, mid-merge failure handling, and per-page access records.
"""

import random
import unittest

from rankmerge import Source, topk


# ---------------------------------------------------------------------------
# Independent references and test doubles
# ---------------------------------------------------------------------------

def brute_force(tables, k):
    """Independent full-materialisation reference over plain dicts."""
    totals = {}
    for table in tables:
        for ident, score in table.items():
            totals[ident] = totals.get(ident, 0) + score
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k]


def reference_topk(sources, k, page_size):
    """Independent reference using only the public handle API, reading every
    page (no early stopping)."""
    handles = [s.open() for s in sources]
    try:
        totals = {}
        for handle in handles:
            cursor = None
            while True:
                items, cursor = handle.page(cursor, page_size)
                for ident, score in items:
                    totals[ident] = totals.get(ident, 0) + score
                if cursor is None:
                    break
        return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    finally:
        for handle in handles:
            handle.close()


class RecordingHandle:
    """Handle test double: forwards through the public handle API only and
    records every call.  It has no ``_items``/``_scores`` of its own, so a
    merge that touched private handle state would fail loudly."""

    def __init__(self, owner, inner):
        self._owner = owner
        self._inner = inner
        self.closed = False

    def page(self, cursor, n):
        owner = self._owner
        owner.events.append(("page", cursor, n))
        if owner.fail_on_page is not None:
            owner.fail_on_page -= 1
            if owner.fail_on_page == 0:
                raise RuntimeError("injected page failure")
        items, nxt = self._inner.page(cursor, n)
        owner.events.append(("items", tuple(items), nxt))
        if owner.after_page is not None:
            owner.after_page(owner)
        return items, nxt

    def lookup(self, ids):
        ids = list(ids)
        self._owner.events.append(("lookup", tuple(ids)))
        if self._owner.fail_on_lookup:
            raise RuntimeError("injected lookup failure")
        return self._inner.lookup(ids)

    def close(self):
        self.closed = True
        self._owner.events.append(("close",))
        self._inner.close()


class RecordingSource:
    """Source test double wrapping a real Source via its public API."""

    def __init__(self, source):
        self._source = source
        self.events = []
        self.handles = []
        self.fail_on_page = None   # countdown; the page call reaching 0 raises
        self.fail_on_lookup = False
        self.fail_on_open = False
        self.after_page = None     # optional callback(owner) after each page

    def open(self):
        self.events.append(("open",))
        if self.fail_on_open:
            raise RuntimeError("injected open failure")
        handle = RecordingHandle(self, self._source.open())
        self.handles.append(handle)
        return handle

    def page_events(self):
        return [e for e in self.events if e[0] == "page"]

    def lookup_events(self):
        return [e for e in self.events if e[0] == "lookup"]


# ---------------------------------------------------------------------------
# Source / handle contract
# ---------------------------------------------------------------------------

class SourceContractTests(unittest.TestCase):
    def test_construction_validation(self):
        with self.assertRaises(ValueError):
            Source({"": 1})                      # empty id
        with self.assertRaises(ValueError):
            Source({"a": -1})                    # negative score
        with self.assertRaises(TypeError):
            Source({"a": 1.5})                   # non-int score
        with self.assertRaises(TypeError):
            Source({"a": True})                  # bool is not an int score
        with self.assertRaises(TypeError):
            Source({7: 1})                       # non-str id
        with self.assertRaises(TypeError):
            Source([("a", 1)])                   # not a mapping
        src = Source({"a": 1})
        self.assertEqual(src.version, 0)

    def test_replace_bumps_version(self):
        src = Source({"a": 1})
        self.assertEqual(src.replace({"b": 2}), 1)
        self.assertEqual(src.version, 1)
        handle = src.open()
        self.assertEqual(handle.version, 1)
        self.assertEqual(handle.page(None, 10), ([("b", 2)], None))
        self.assertEqual(src.replace({}), 2)
        self.assertEqual(src.version, 2)
        with self.assertRaises(ValueError):
            src.replace({"x": -1})               # failed replace keeps state
        self.assertEqual(src.version, 2)

    def test_open_freezes_independent_snapshot(self):
        src = Source({"a": 5, "b": 3})
        handle = src.open()
        src.replace({"c": 99})
        # The frozen snapshot still serves the old version.
        self.assertEqual(handle.version, 0)
        self.assertEqual(handle.page(None, 10), ([("a", 5), ("b", 3)], None))
        self.assertEqual(handle.lookup(["a", "c"]), {"a": 5, "c": 0})
        # A fresh handle sees the new version.
        fresh = src.open()
        self.assertEqual(fresh.version, 1)
        self.assertEqual(fresh.page(None, 10), ([("c", 99)], None))

    def test_page_ordering_and_cursor_chain(self):
        table = {"d": 5, "b": 5, "a": 5, "c": 9, "e": 1}
        handle = Source(table).open()
        # Score descending, id ascending for ties.
        self.assertEqual(
            handle.page(None, 2), ([("c", 9), ("a", 5)], 2)
        )
        self.assertEqual(handle.page(2, 2), ([("b", 5), ("d", 5)], 4))
        self.assertEqual(handle.page(4, 2), ([("e", 1)], None))  # end: None
        # One big page reads everything and reports the end.
        self.assertEqual(
            Source(table).open().page(None, 99),
            ([("c", 9), ("a", 5), ("b", 5), ("d", 5), ("e", 1)], None),
        )
        # Empty snapshot: immediately at the end.
        self.assertEqual(Source({}).open().page(None, 3), ([], None))

    def test_page_argument_validation(self):
        handle = Source({"a": 1}).open()
        with self.assertRaises(ValueError):
            handle.page(None, 0)
        with self.assertRaises(ValueError):
            handle.page(None, -2)
        with self.assertRaises(ValueError):
            handle.page("bogus", 1)
        with self.assertRaises(ValueError):
            handle.page(5, 1)                    # cursor out of range

    def test_lookup_missing_is_zero(self):
        handle = Source({"a": 4, "b": 0}).open()
        self.assertEqual(
            handle.lookup(["a", "missing", "b"]),
            {"a": 4, "missing": 0, "b": 0},
        )
        self.assertEqual(handle.lookup([]), {})

    def test_closed_handle_raises(self):
        handle = Source({"a": 1}).open()
        handle.close()
        handle.close()                           # close is idempotent
        self.assertTrue(handle.closed)
        with self.assertRaises(RuntimeError):
            handle.page(None, 1)
        with self.assertRaises(RuntimeError):
            handle.lookup(["a"])


# ---------------------------------------------------------------------------
# topk behaviour
# ---------------------------------------------------------------------------

class TopkTests(unittest.TestCase):
    def test_basic_merge(self):
        a = Source({"x": 10, "y": 7})
        b = Source({"x": 1, "z": 9})
        self.assertEqual(topk([a, b], 2, 1), [("x", 11), ("z", 9)])
        # The same source may appear twice; scores add up.
        self.assertEqual(topk([a, a], 1, 1), [("x", 20)])

    def test_ties_broken_by_id(self):
        src = Source({"b": 5, "a": 5, "c": 5})
        self.assertEqual(topk([src], 3, 1), [("a", 5), ("b", 5), ("c", 5)])
        self.assertEqual(topk([src], 1, 1), [("a", 5)])
        # Cross-source tie at the k boundary.
        a = Source({"x": 3, "y": 3})
        b = Source({"x": 2, "y": 2})
        self.assertEqual(topk([a, b], 1, 1), [("x", 5)])

    def test_tie_with_unseen_id_keeps_reading(self):
        # After page 1 ("b", 5) an unseen id could still total 5 with a
        # lexicographically smaller id, so stopping there would be wrong.
        rec = RecordingSource(Source({"b": 5, "a": 5, "c": 4}))
        got = topk([rec], 1, 1)
        self.assertEqual(got, [("a", 5)])
        # Stop became legal only once the running score dropped below 5.
        self.assertEqual(len(rec.page_events()), 3)

    def test_empty_sources(self):
        self.assertEqual(topk([Source({}), Source({})], 3, 2), [])
        self.assertEqual(topk([Source({}), Source({"a": 2})], 3, 2), [("a", 2)])
        self.assertEqual(topk([], 3, 2), [])
        self.assertEqual(topk([Source({"a": 2})], 0, 2), [])
        # k larger than the id universe returns everything that exists.
        self.assertEqual(
            topk([Source({"a": 2, "b": 1})], 10, 1), [("a", 2), ("b", 1)]
        )
        # Zero scores are legal and such ids still exist.
        self.assertEqual(topk([Source({"a": 0}), Source({"b": 0})], 1, 1),
                         [("a", 0)])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            topk([Source({})], -1, 1)
        with self.assertRaises(ValueError):
            topk([Source({})], 1, 0)

    def test_handles_closed_and_sources_left_open(self):
        real = [Source({"a": 1, "b": 2}), Source({"b": 3, "c": 4})]
        recs = [RecordingSource(s) for s in real]
        got = topk(recs, 2, 1)
        self.assertEqual(got, [("b", 5), ("c", 4)])
        for rec in recs:
            self.assertEqual(len(rec.handles), 1)
            self.assertTrue(rec.handles[0].closed)
            self.assertEqual(
                [e[0] for e in rec.events].count("close"), 1
            )
        # The caller's sources were not closed or otherwise disturbed.
        self.assertEqual([s.version for s in real], [0, 0])
        for s in real:
            s.open().close()

    def test_replace_interleaved_with_merge_keeps_snapshot(self):
        a = Source({"x": 50, "y": 40, "z": 30, "w": 20})
        b = Source({"p": 100, "x": 5, "y": 5, "q": 1})
        original = [
            {"x": 50, "y": 40, "z": 30, "w": 20},
            {"p": 100, "x": 5, "y": 5, "q": 1},
        ]
        ra, rb = RecordingSource(a), RecordingSource(b)
        fired = []

        def after_page(owner):
            if not fired:
                fired.append(True)
                a.replace({"x": 1000})
                b.replace({"p": 0})

        ra.after_page = after_page
        rb.after_page = after_page
        got = topk([ra, rb], 3, 2)
        # The merge ran on the frozen snapshots despite the mid-read replace.
        self.assertTrue(fired)
        self.assertEqual(got, [("p", 100), ("x", 55), ("y", 45)])
        self.assertEqual(got, brute_force(original, 3))
        self.assertEqual((a.version, b.version), (1, 1))
        # A fresh merge sees the replaced tables.
        self.assertEqual(topk([a, b], 3, 2), [("x", 1000), ("p", 0)])

    def test_page_failure_closes_every_open_handle(self):
        real = [
            Source({"a": 9, "b": 8, "c": 7}),
            Source({"d": 9, "e": 8, "f": 7}),
            Source({"g": 9, "h": 8, "i": 7}),
        ]
        recs = [RecordingSource(s) for s in real]
        recs[1].fail_on_page = 1                 # first page of source 2 raises
        with self.assertRaisesRegex(RuntimeError, "injected page failure"):
            topk(recs, 2, 1)
        for rec in recs:
            self.assertEqual(len(rec.handles), 1)
            self.assertTrue(rec.handles[0].closed)
        # Sources themselves remain usable.
        for s in real:
            s.open().close()

    def test_lookup_failure_closes_every_open_handle(self):
        # Winner "x" is only seen in source 1, so completion needs a lookup
        # on source 2 -- which is made to fail.
        real = [Source({"x": 100, "a1": 3}), Source({"b1": 2, "b2": 1})]
        recs = [RecordingSource(s) for s in real]
        recs[1].fail_on_lookup = True
        with self.assertRaisesRegex(RuntimeError, "injected lookup failure"):
            topk(recs, 1, 1)
        for rec in recs:
            self.assertTrue(rec.handles[0].closed)

    def test_open_failure_closes_already_open_handles(self):
        real = [Source({"a": 1}), Source({"b": 2}), Source({"c": 3})]
        recs = [RecordingSource(s) for s in real]
        recs[1].fail_on_open = True
        with self.assertRaisesRegex(RuntimeError, "injected open failure"):
            topk(recs, 2, 1)
        self.assertTrue(recs[0].handles[0].closed)   # opened, then closed
        self.assertEqual(recs[1].handles, [])        # open raised
        self.assertEqual(recs[2].handles, [])        # never reached


# ---------------------------------------------------------------------------
# Page-access records: early stopping and on-demand reads
# ---------------------------------------------------------------------------

class PageAccessTests(unittest.TestCase):
    def test_early_stop_and_page_records(self):
        tail = {"d%02d" % i: 79 - i for i in range(20)}   # 79..60
        table = dict({"x": 100, "y": 90, "z": 80}, **tail)
        ra = RecordingSource(Source(table))
        rb = RecordingSource(Source(dict(table)))
        got = topk([ra, rb], 3, 3)
        self.assertEqual(got, [("x", 200), ("y", 180), ("z", 160)])

        pages_a, pages_b = ra.page_events(), rb.page_events()
        # Bound analysis stops after A1, B1, A2: after B1 the threshold
        # 80+80 ties z's 160 (an unseen smaller id could still displace z);
        # after A2 the threshold is 77+80=157 < 160 and every challenger's
        # upper bound is below 160.
        self.assertEqual(len(pages_a) + len(pages_b), 3)
        self.assertEqual([e[1] for e in pages_a], [None, 3])  # cursor chain
        self.assertEqual([e[1] for e in pages_b], [None])
        self.assertTrue(all(e[2] == 3 for e in pages_a + pages_b))  # n
        # Winners were fully seen in both sources: no lookup was needed.
        self.assertEqual(ra.lookup_events(), [])
        self.assertEqual(rb.lookup_events(), [])
        # A full scan would have needed ceil(23/3)=8 pages per source.
        self.assertLess(3, 8 * 2)
        for rec in (ra, rb):
            self.assertTrue(rec.handles[0].closed)

    def test_lookup_completes_winner_unseen_in_a_source(self):
        ra = RecordingSource(Source({"x": 100, "a1": 3}))
        rb = RecordingSource(Source({"b1": 2, "b2": 1}))
        got = topk([ra, rb], 1, 1)
        self.assertEqual(got, [("x", 100)])
        # "x" never appeared in source 2's pages, so its exact total was
        # completed with a single random-access lookup (missing -> 0).
        self.assertEqual(ra.lookup_events(), [])
        self.assertEqual(rb.lookup_events(), [("lookup", ("x",))])

    def test_no_extra_pages_after_stream_end(self):
        rec = RecordingSource(Source({"a": 2, "b": 1}))
        got = topk([rec], 5, 1)
        self.assertEqual(got, [("a", 2), ("b", 1)])
        pages = rec.page_events()
        self.assertEqual(len(pages), 2)          # 2 items, then stream ended
        results = [e for e in rec.events if e[0] == "items"]
        self.assertIsNone(results[-1][2])        # last page reported None


# ---------------------------------------------------------------------------
# Differential test against the independent full references
# ---------------------------------------------------------------------------

class DifferentialTests(unittest.TestCase):
    def test_against_full_references(self):
        rng = random.Random(20261115)
        id_pool = [chr(ord("a") + i) for i in range(10)] + ["aa", "ab", "zz"]
        for trial in range(400):
            m = rng.randint(1, 4)
            tables = []
            for _ in range(m):
                table = {}
                for ident in id_pool:
                    if rng.random() < 0.6:
                        table[ident] = rng.randint(0, 6)   # small range: ties
                tables.append(table)
            k = rng.randint(0, 12)
            page_size = rng.randint(1, 20)
            recs = [RecordingSource(Source(t)) for t in tables]
            got = topk(recs, k, page_size)
            want = brute_force(tables, k)
            self.assertEqual(
                got, want,
                "trial %d: tables=%r k=%r page_size=%r" % (trial, tables, k, page_size),
            )
            if trial % 50 == 0:  # spot-check the handle-based reference too
                self.assertEqual(
                    got, reference_topk([Source(t) for t in tables], k, page_size)
                )
            # Per-page access-record invariants for every handle.
            for rec, table in zip(recs, tables):
                pages = rec.page_events()
                results = [e for e in rec.events if e[0] == "items"]
                self.assertEqual(len(pages), len(results))
                expect_cursor = None
                for (_, cursor, n), (_, items, nxt) in zip(pages, results):
                    self.assertEqual(cursor, expect_cursor)  # chained cursors
                    self.assertEqual(n, page_size)
                    self.assertLessEqual(len(items), page_size)
                    expect_cursor = nxt
                # No page was fetched after a page reported the end.
                for _, items, nxt in results[:-1]:
                    self.assertIsNotNone(nxt)
                # Never more pages than a full scan would need.
                full_scan = max(1, -(-len(table) // page_size))
                self.assertLessEqual(len(pages), full_scan)
                # Exactly one open and one close per source.
                kinds = [e[0] for e in rec.events]
                self.assertEqual(kinds.count("open"), 1)
                self.assertEqual(kinds.count("close"), 1)
                self.assertTrue(rec.handles[0].closed)


if __name__ == "__main__":
    unittest.main()
