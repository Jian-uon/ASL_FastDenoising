# docs/ — index

All documentation for **ASL_FastDenoising** (accelerated 7T ASL perfusion denoising with T1
cross-attention guidance). Start from [../CLAUDE.md](../CLAUDE.md); this index says what else exists.

## Active

| Document | Content |
|---|---|
| `v35_paper/` ⛔ | **Local only — not in this public repo.** The paper line: `README.md` (scope + isolation rules), `experiment_plan.md` (figures F1–F7, tables T1–T5, training matrix, 5-week roadmap), `ccr2026_abstract_en.txt` (submitted 2026-08-25). Kept out because the unrun experiment matrix and the submission target are the parts a competitor could act on. |
| [validation_metrics.md](validation_metrics.md) | Metric definitions (uMSE/uPSNR, psnr_b, CNR, sCoV, EFC, lapvar, hfen, gmsd, SURE) and the selection philosophy. **Load-bearing** — read before touching `--best_criterion`. |
| [v37_legacy.md](v37_legacy.md) | Full architecture + tensor-dimension flow of this conv family (ConvEncoder2D, multi-scale cross-attention, tissue bias, T1-free detail decoder, SVFW, stage protocol). The most complete structural reference for the current model. |
| `v35_patent_version.md` ⛔ | **Local only — not in this repo.** V35 design/status record: exact flags, restored config, loss recipe, the two historical launch faults, and the known-bug list the code-fix backlog is built on. Purged from git history 2026-08-26 (it carries the patent-filing decision). |
| `patent_draft.md` ⛔ | **Local only — not in this repo.** Patent draft, purged from git history 2026-08-26: the patent is unfiled and this repo is public, so committing it would destroy novelty. ⚠ It is written against v37 with **SVFW as claim #1**; V35 does not use SVFW, so filing on this line requires promoting the multi-scale cross-attention to claim #1. |
| `related_work.md` ⛔ | **Local only — not in this public repo.** Prior-art survey for this method family + ASL-denoising competitor table (Shou, Guo, Gong, Xie), SURE literature, BLUE/Set-Transformer lineage. |
| `publishability_precedents.md` ⛔ | **Local only — not in this public repo.** 29 verified published precedents (ASL DL denoising, self-supervised no-clean-GT MRI, accelerated-MRI validation practice). |

## Figures

| File | Use |
|---|---|
| [figures/architecture_v37.tex](figures/architecture_v37.tex) | Editable TikZ block diagram of the conv pipeline — the starting point for the paper's F1 schematic. |
| [figures/fra_block.svg](figures/fra_block.svg) / `.png` | Frame Reliability Aggregator internals. ⚠ **The input is annotated `C=5` (a 5-slice z-window)** — that belongs to the other repo's 2.5-D backbone. This line is 2-D: **relabel to `C=1` before use.** |
| `figures/asl_principle_challenges_nature.{svg,pdf,png}` + [contract](figures/asl_principle_challenges_nature_contract.md) / [qa](figures/asl_principle_challenges_nature_qa.md) | ASL principle + few-frame problem motivation figure (method-agnostic). ⚠ **Was authored as the CIG-VSS/KBS manuscript's Figure 1.** Reusing it verbatim in both papers is a duplicate-figure risk — regenerate a visually distinct variant from [../scripts/figures/make_asl_principle_challenges_nature.py](../scripts/figures/make_asl_principle_challenges_nature.py) before submission. |
| `figures/sure_*.png` | Illustrations for [archive/brainstorm/sure_explained.md](archive/brainstorm/sure_explained.md). |

## Archive

Historical material carried over from the source repository, kept because it documents *this*
method family's negative results and earlier plans:

- [archive/history/v42i_drop_svfw.md](archive/history/v42i_drop_svfw.md) — **why the per-pixel SVFW aggregator was dropped** in favour of the scalar FRA (probe: SVFW amplified per-pixel noise, uniform-mean replacement cut lapvar 49%). Load-bearing whenever anyone proposes reviving per-pixel frame weighting.
- [archive/history/experiments_plan.md](archive/history/experiments_plan.md) — the earlier BSPC/CMIG experiment ledger for the same conv method; predecessor of `v35_paper/experiment_plan.md` (that successor is local only).
- [archive/paper_drafts/paper_draft.md](archive/paper_drafts/paper_draft.md) — an earlier BSPC-targeted draft of this line ("SVFW-Net"). Its intro, gap statement and cross-modal-transformer critique are reusable prose.
- [archive/brainstorm/sure_explained.md](archive/brainstorm/sure_explained.md) — SURE math + the divergence-estimate bug record (the loss still uses `w_sure 0.02`).

> **Note on archive links.** These files came from the source repo and some of their internal
> links point at documents that were deliberately **not** migrated (CIG-VSS design docs, that
> line's paper materials, retired-module notes). The prose is still valid; only the links dangle.
> Active documents (the table above) have been checked and are link-clean.
