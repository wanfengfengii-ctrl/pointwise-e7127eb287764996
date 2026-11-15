"""Demo: merge the top-K of several independent scoring sources."""

from rankmerge import Source, topk


def main():
    # Three independent scoring sources (e.g. three ranking services).
    search = Source({"apple": 91, "banana": 88, "cherry": 70, "date": 12})
    clicks = Source({"banana": 95, "cherry": 80, "elderberry": 60})
    fresh = Source({"apple": 10, "cherry": 99, "fig": 5})
    sources = [search, clicks, fresh]

    print("top 3 merged ranking (total desc, id asc):")
    for rank, (ident, total) in enumerate(topk(sources, k=3, page_size=2), 1):
        print("  %d. %-10s %d" % (rank, ident, total))

    # A snapshot stays frozen while the source moves on.
    handle = search.open()
    search.replace({"kiwi": 100})
    print("\n'search' replaced to version %d, but the open snapshot (version %d) still serves:" % (search.version, handle.version))
    print(" ", handle.page(None, 10)[0])
    handle.close()

    fresh_handle = search.open()
    print("fresh snapshot of 'search' (version %d):" % fresh_handle.version)
    print(" ", fresh_handle.page(None, 10)[0])
    fresh_handle.close()

    print("\ntop 2 after the replace:")
    for rank, (ident, total) in enumerate(topk(sources, k=2, page_size=2), 1):
        print("  %d. %-10s %d" % (rank, ident, total))


if __name__ == "__main__":
    main()
