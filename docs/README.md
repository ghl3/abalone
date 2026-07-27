# docs/

Dated engineering documents: reviews, design notes, post-mortems, roadmaps.

## Naming convention

```
YYYY-MM-DD-<kebab-case-slug>.md
```

- The date is the date the document was **written**, not revised. Documents are
  point-in-time records; if the conclusions change, write a new one and link
  back rather than editing history.
- Filenames sort chronologically in a plain `ls`, so `ls docs/` is the timeline.
- Keep the slug short and searchable (`architecture-review`, `move-encoding`,
  `postmortem-gen3-collapse`).

Long-lived reference material that is *not* a point-in-time record (a spec, a
design doc) goes in `docs/` under an `UPPERCASE.md` name, without a date prefix.
These are living documents and are edited in place.

## Index

### Living documents

| Document | Summary |
| --- | --- |
| [MODEL.md](MODEL.md) | Target design for the network and its training. Heuristic-free: score-based termination, capture-handicap curriculum, deep batched MCTS, convolutional policy head, auxiliary heads. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Target system architecture across Rust, Python and the web UI. Component responsibilities, the generation cycle, cross-language contracts, storage layout, invariants. |

### Dated records

| Document | Summary |
| --- | --- |
| [2026-07-27-architecture-review.md](2026-07-27-architecture-review.md) | Full-codebase review + roadmap. Diagnoses why the gen 1–3 training run learned nothing; two silent correctness bugs; prioritised plan. |
