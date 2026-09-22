# Retrieval tool overhead repair

Date: 2026-09-14 UTC. Scope: tool implementation, development replay and regression
verification. The old core experiment and heartbeat remain paused. This work
does not register or start a replacement Agent quality experiment.

## Findings and changes

The original BrowseComp literal scan timed out after reading about 40,000 of
100,195 documents in the profiler. Reading files accounted for most of that
time; complete-record construction also calculated revision/identity hashes for
documents that never matched. Regex matching rebuilt separate normalized files
on each request and timed out before reaching rg. Removing temporary files then
added approximately 3.5 seconds beyond the call deadline in the measured case.

Exact matching now keeps a disposable cache owned by the query session:

- A packed file stores normalized UTF-8 bodies, avoiding 100,000 separate file
  opens per literal query. Returned documents alone acquire revision hashes.
- File device/inode/size/mtime/ctime signatures are rechecked on every scan.
  Updates refresh cached text; complete scans remove deleted, renamed or excluded
  sources. Direct reads remain live. Symlink confinement and source validation
  are retained. Interrupted preparation can continue without treating partial
  cached data as a successful corpus preparation.
- Regex and case folding reuse normalized files. rg scans with four CPU threads;
  ARKB explicitly restores source/position ordering. Filename discovery precedes
  bounded batches of matching-text JSON, avoiding full-corpus text serialization
  for broad patterns. This can use a discovery process plus span batches instead
  of the old single rg process; it is not one process per document.
- Timeouts and cancellation include source checks and contention for the cache.
  Runtime closes packed files and removes disposable files, including on failure.

`DocumentAccess` enumerates the flat scope with `scandir`, sorts filename strings,
and resolves a supplied source directly. It no longer resolves every ancestor or
sorts full Path objects for a source-filtered read.

BM25's common-term query matched 1,714,606 chunks. In the original profile,
approximately 36 of 38 seconds were spent creating sort keys, including millions
of JSON/SHA operations. BM25 now computes immutable tie identities during engine
preparation and selects only the requested top K with a heap. Tokenization,
statistics, floating-point scoring order, source filtering and deterministic
tie rules are unchanged. Hybrid uses this same BM25 instance; its semantic leg,
candidate depth and fusion algorithm are unchanged.

## Preparation is explicit

Large-scope callers should prepare tools before starting measured queries:

```python
tools = runtime.agent_tools(
    engine=engine, directory=corpus, vault_id=vault_id, prepare_exact=True,
)
```

The development Agentic runner now requests this preparation inside its existing
setup phase. Its frozen historical copy was not edited. General callers retain
lazy preparation unless they opt in; a cold full-corpus scan can still exhaust
a short deadline. The repaired cache does not make initial corpus reading free.

Preparing all 100,195 BrowseComp documents took approximately 54 seconds in the
final replay. Its normalized text is 3.025 GiB; preparing both packed text and rg
files uses about 6.05 GiB of logical scratch content, plus filesystem overhead,
filename matching files and space for edits. The cache lives for the Runtime or
explicit ExactRetriever context; it is not a new permanent index. BM25 setup was
335.9 seconds before and 344.1 seconds after in the measured runs. Report these
costs separately and amortize them only when the same prepared tools are reused.

No documents, chunks, vectors or source corpus files were rewritten or removed.
No embedding regeneration, model change or benchmark corpus reduction was used.

## Verification and measured costs

All three full-index BM25 probes produced exactly the same serialized response
hash before and after the repair, including scores and evidence metadata:

| Probe | Original, with cProfile | Repaired, with cProfile |
| --- | ---: | ---: |
| Common term | 38.186 s | 0.580 s |
| Multiple terms | 37.009 s | 0.882 s |
| Absent term | 0.000095 s | 0.000074 s |

These are instrumented diagnostic timings. Profiler overhead depends on function
call counts, so their ratios are not promises of production speedup.

The final prepared BrowseComp probes, without cProfile, took 2.85 seconds for an
absent literal, 4.36 seconds for an absent regex, 0.076 seconds for a frequent
literal and 6.86 seconds for the same frequent regex. Literal and regex results
matched exactly on these shared patterns. The original absent probes exceeded
their 30-second limits; that is a timeout lower bound, not a measured completed
baseline with which to compute an exact speedup.

The replay covers every executed match event in the existing pilot, including
repeated patterns and all original failures. It does not choose calls by their
success or inspect answer labels. Successful historical evidence is compared
field by field; returned slices and revisions are checked against live files.
The final replay passed all **87 calls**, including all **25 prior timeouts**.
All **62 previously successful responses** were preserved, and **85 returned
references** matched the live source text and revision. BrowseComp's 32 calls
averaged **4.10 seconds**, with **6.17-second p95** and no timeouts.
See the [final replay summary](../evaluation/agentic-tools/tool-overhead-v1/replay-final/summary.json)
and the [artifact guide](../evaluation/agentic-tools/tool-overhead-v1/README.md).

The project and Agentic harness regression run passed **1,251 tests**, with 60
integration-marked cases excluded. Added coverage includes same-size edits with
restored mtime, rename/delete/filter ordering, symlink replacement, invalid source
names, interrupted preparation, partial writes, cleanup and lock-wait timeouts.
BM25 is checked against full-sort reference scoring with dense ties, source
filters, reversed corpus order and several K values. Existing Hybrid tests pass.
An additional **252 differential checks** against the frozen original exact
implementation passed, covering empty scopes, zero-width and multiline patterns,
Unicode/case folding, invalid regex, filename filters and three result limits.

The initial bare pytest discovery traversed an archived MLX probe dependency and
aborted during collection. The successful run explicitly targeted `tests` and
`evaluation/agentic_tools`; it did not import archived third-party test suites.

The replay uses exposed development tool calls, not independent answer-quality
evidence. Another user training workflow may share the machine; these timings
are diagnostic, not isolated latency measurements. Live GPU Hybrid timing and
Agent answer generation were not part of this repair validation. The unchanged
Hybrid composition is covered by regression tests and the exact full-index BM25
response comparison, not a newly claimed end-to-end Hybrid benchmark.

## Next evaluation boundary

The 90-file frozen core source snapshot remains unchanged, and the suspended
11 completed / one interrupted / 5,448 unstarted attempts remain separate.
Do not restart that protocol with the repaired source. The next study still
needs its revised tool-level dispatch gate, justified sample/repetition and
execution policy, a new source/protocol identity and the full registered pilot.
This repair does not establish which Agent strategy gives better answers.
