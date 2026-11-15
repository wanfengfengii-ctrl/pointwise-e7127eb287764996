"""rankmerge -- merge the top-K of several independent scoring sources.

Public API (standard library only):

* ``Source(table)``      -- a versioned table mapping non-empty string IDs to
                            non-negative integer scores.  Starts at version 0.
* ``Source.replace(t)``  -- atomically replace the whole table, bump version.
* ``Source.open()``      -- freeze the current version into an independent
                            snapshot handle.
* ``handle.page(c, n)``  -- sorted access: at most ``n`` ``(id, score)`` pairs
                            ordered by score descending, then id ascending,
                            plus the next cursor.  The first cursor is
                            ``None``; the end of the stream is reported as a
                            ``None`` next cursor.
* ``handle.lookup(ids)`` -- random access: ``{id: score}`` with 0 for ids
                            absent from the snapshot.
* ``handle.close()``     -- release the snapshot; ``page``/``lookup`` on a
                            closed handle raise ``RuntimeError``.
* ``topk(sources, k, page_size)`` -- merged top-k as ``[(id, total), ...]``
                            ordered by total score descending, then id
                            ascending.  Only ids present in at least one
                            source are eligible.
"""

from collections.abc import Mapping

__all__ = ["Source", "SnapshotHandle", "topk"]


def _checked_table(table, what):
    """Validate the ``str -> int`` contract and return a plain dict copy."""
    if not isinstance(table, Mapping):
        raise TypeError(
            "%s must be a mapping of str -> int, got %s"
            % (what, type(table).__name__)
        )
    out = {}
    for ident, score in table.items():
        if not isinstance(ident, str):
            raise TypeError(
                "%s id must be a str, got %s" % (what, type(ident).__name__)
            )
        if ident == "":
            raise ValueError("%s id must be a non-empty string" % what)
        if isinstance(score, bool) or not isinstance(score, int):
            raise TypeError(
                "%s score for %r must be an int" % (what, ident)
            )
        if score < 0:
            raise ValueError(
                "%s score for %r must be non-negative" % (what, ident)
            )
        out[ident] = score
    return out


class Source:
    """A versioned, replaceable score table."""

    __slots__ = ("_table", "_version")

    def __init__(self, table):
        self._table = _checked_table(table, "source")
        self._version = 0

    @property
    def version(self):
        """Current version: 0 at construction, +1 per ``replace``."""
        return self._version

    def replace(self, table):
        """Atomically replace the whole table and bump the version."""
        self._table = _checked_table(table, "replacement")
        self._version += 1
        return self._version

    def open(self):
        """Freeze the current version into an independent snapshot handle."""
        return SnapshotHandle(self._table, self._version)

    def __repr__(self):
        return "Source(version=%r, size=%r)" % (self._version, len(self._table))


class SnapshotHandle:
    """An immutable, independent view of one version of a ``Source``.

    The cursor is an opaque token: pass ``None`` for the first page and then
    whatever the previous ``page`` call returned.
    """

    __slots__ = ("_items", "_scores", "_version", "_closed")

    def __init__(self, table, version):
        # Independent frozen copy: later replace() calls on the source cannot
        # disturb this snapshot.
        self._items = sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))
        self._scores = dict(table)
        self._version = version
        self._closed = False

    @property
    def version(self):
        """The source version this snapshot was frozen from."""
        return self._version

    @property
    def closed(self):
        return self._closed

    def _require_open(self):
        if self._closed:
            raise RuntimeError("snapshot handle is closed")

    def page(self, cursor, n):
        """Return ``(items, next_cursor)`` for the page starting at ``cursor``.

        ``items`` is a list of at most ``n`` ``(id, score)`` pairs ordered by
        score descending, then id ascending.  ``next_cursor`` is ``None`` once
        the stream is exhausted; ``cursor=None`` reads the first page.
        """
        self._require_open()
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("page size n must be a positive integer")
        size = len(self._items)
        if cursor is None:
            start = 0
        elif (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or not 0 <= cursor < size
        ):
            raise ValueError(
                "invalid cursor %r: use None or a token from a previous page()"
                % (cursor,)
            )
        else:
            start = cursor
        stop = min(start + n, size)
        nxt = stop if stop < size else None
        return list(self._items[start:stop]), nxt

    def lookup(self, ids):
        """Return ``{id: score}`` for the requested ids; missing ids score 0."""
        self._require_open()
        scores = self._scores
        return {ident: scores.get(ident, 0) for ident in ids}

    def close(self):
        """Release the snapshot.  Idempotent; reads after close raise."""
        self._items = []
        self._scores = {}
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def topk(sources, k, page_size):
    """Merge the top-k of *sources* into ``[(id, total_score), ...]``.

    One snapshot handle per source is opened up front; the merge then works
    purely through those public handles (``page``/``lookup``/``close``).  The
    caller's ``Source`` objects are never closed and may be ``replace``d at
    any moment without affecting the in-flight merge.

    After every page read, legal score bounds are recomputed: each seen id
    has a lower bound (sum of its known per-source scores) and an upper
    bound (known scores plus, per source where it is still unseen, the best
    score that source can still yield); every never-seen id shares the
    threshold bound (sum of those per-source remainders).  Reading stops as
    soon as at least ``k`` candidates are determined and no partially-seen or
    never-seen id can possibly reach the top-k -- ties included, because an
    unseen id could be lexicographically smaller than any known id, so the
    threshold must drop strictly below the k-th lower bound.  The winners'
    exact totals are then completed with ``lookup`` where needed.

    All opened handles are closed again, whether the merge returns or raises.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
        raise ValueError("page_size must be a positive integer")
    handles = []
    try:
        for source in sources:
            handles.append(source.open())
        if k == 0 or not handles:
            return []
        return _merge_topk(handles, k, page_size)
    finally:
        for handle in handles:
            handle.close()


def _merge_topk(handles, k, page_size):
    m = len(handles)
    cursors = [None] * m        # next cursor per source (None = not started)
    started = [False] * m       # at least one page fetched
    exhausted = [False] * m     # stream finished (a page returned None cursor)
    last_score = [0] * m        # smallest score yielded so far per source
    seen = {}                   # id -> list of m slots (score or None)

    def remaining_bounds():
        # Best score source i can still yield for an id not yet seen from it.
        bounds = []
        for i in range(m):
            if exhausted[i]:
                bounds.append(0)
            elif not started[i]:
                bounds.append(None)  # unbounded: source not read yet
            else:
                bounds.append(last_score[i])
        return bounds

    def complete_winners(winners):
        # Fill in exact totals for the winning ids via lookup where a source
        # has not yielded them yet, then rank exactly.
        ids = [ident for _, _, ident in winners]
        totals = {ident: 0 for ident in ids}
        for i in range(m):
            missing = [ident for ident in ids if seen[ident][i] is None]
            if missing:
                got = handles[i].lookup(missing)
                for ident in missing:
                    totals[ident] += got[ident]
            for ident in ids:
                score = seen[ident][i]
                if score is not None:
                    totals[ident] += score
        return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))

    def try_stop():
        """Return the final top-k list, or None if more reading is needed."""
        if len(seen) < k:
            return None
        bounds = remaining_bounds()
        if any(b is None for b in bounds):
            return None
        threshold = sum(bounds)  # best possible total of a never-seen id
        entries = []  # (lower, upper, id) per seen id
        for ident, slots in seen.items():
            lower = 0
            slack = 0
            for i in range(m):
                score = slots[i]
                if score is None:
                    slack += bounds[i]
                else:
                    lower += score
            entries.append((lower, lower + slack, ident))
        entries.sort(key=lambda e: (-e[0], e[2]))
        winners = entries[:k]
        worst_lower, _, worst_id = winners[-1]
        for _, upper, ident in entries[k:]:
            # A challenger displaces the k-th candidate iff its final total
            # can exceed worst_lower, or tie it with a smaller id.
            if upper > worst_lower or (upper == worst_lower and ident < worst_id):
                return None
        # A never-seen id may have any (smaller) id, so its bound must drop
        # strictly below the k-th lower bound.
        if threshold >= worst_lower:
            return None
        return complete_winners(winners)

    while True:
        if all(exhausted):
            # Every existing id has been seen; all totals are exact.
            totals = {}
            for ident, slots in seen.items():
                totals[ident] = sum(s for s in slots if s is not None)
            ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
            return ranked[:k]
        for i in range(m):
            if exhausted[i]:
                continue
            items, nxt = handles[i].page(cursors[i], page_size)
            if not items and nxt is not None:
                raise RuntimeError(
                    "snapshot handle returned an empty page before end of stream"
                )
            started[i] = True
            for ident, score in items:
                slots = seen.get(ident)
                if slots is None:
                    slots = [None] * m
                    seen[ident] = slots
                slots[i] = score
            if items:
                last_score[i] = items[-1][1]
            cursors[i] = nxt
            if nxt is None:
                exhausted[i] = True
            result = try_stop()
            if result is not None:
                return result
