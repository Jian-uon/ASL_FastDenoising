# ASL_FastDenoising — Project Guide for Codex

> **This file is a pointer, not a copy.** Read **[`CLAUDE.md`](CLAUDE.md)** — it is the single,
> maintained project guide (background, data pitfalls, architecture, constraints,
> validation/selection policy, open work, repo layout). Everything below is only the orientation
> you need to know *which* documents to open.
>
> **Why a pointer:** this file used to be a full duplicate of the project guide. It silently
> drifted ~6 weeks out of date and ended up describing a retired architecture and a forbidden
> selection rule as if they were current — exactly the failure mode a second copy invites. It was
> reduced to a pointer on 2026-07-16. **Do not re-inline the content here.**

---

## Orientation

This repository owns **one** model line: accelerated 7T ASL perfusion denoising with T1
cross-attention guidance (`--use_t1_cross_fusion`; FRA aggregator + CMF at 16×16/32×32 +
T1-free detail decoder), trained self-supervised (Noise2Noise) with **no clean ground truth** and,
in the default arm, **no labels at all**.

| Need | Read |
|---|---|
| Everything current (start here) | [CLAUDE.md](CLAUDE.md) |
| Paper plan: figures, tables, training matrix, roadmap | [docs/v35_paper/experiment_plan.md](docs/v35_paper/experiment_plan.md) |
| Submitted CCR2026 abstract | [docs/v35_paper/ccr2026_abstract_en.txt](docs/v35_paper/ccr2026_abstract_en.txt) |
| Metric definitions + selection philosophy | [docs/validation_metrics.md](docs/validation_metrics.md) |
| Architecture & tensor-dimension flow of this conv family | [docs/v37_legacy.md](docs/v37_legacy.md) |
| V35 design/status record + known code faults | `docs/v35_patent_version.md` — **local only, not in this repo** (see below) |
| Patent draft (needs a claim-order rewrite — see the V35 record) | `docs/patent_draft.md` — **local only, not in this repo** (see below) |
| Prior art / competitor positioning | [docs/related_work.md](docs/related_work.md) |
| Historical material (may contain stale links) | [docs/archive/](docs/archive/) |

> **The two patent documents are not in this repo.** `docs/patent_draft.md` and
> `docs/v35_patent_version.md` are gitignored and were purged from git history on
> 2026-08-26: the patent is unfiled and this repo is public, so committing them would be a
> novelty-destroying disclosure. They still exist in local working copies — ask the author
> if you need the V35 design/status record or the claim structure.

## Two rules that keep biting

1. **Never select checkpoints on L1 / `psnr_ref` / `psnr_b`.** Use `uMSE` (and confirm with the
   post-hoc selection pass — the in-loop best is step-gated). See CLAUDE.md §4.
2. **This repo was split out of `ASL_dmvae`**, which still hosts a different paper line
   (CIG-VSS + EC-LRDA). Do not import that line's claims, terminology, or figures. See CLAUDE.md §8.
