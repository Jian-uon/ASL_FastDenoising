"""CIG-Net — Content-Isolated Anatomical Guidance for Self-Supervised 7T ASL Denoising.

RETIRED 2026-07-16 — this is the v2/MoSSM paper-facing alias (defaults to
``use_naf_fusion=True`` = NAFDecoder/CORD + MoSSM), both retired. The main line is
CIG-VSS + EC-LRDA; build it directly via ``models.asl_t1_model.ASLT1Denoiser`` with
``--use_vmamba_encoder ... --ec_lrda``. The runner rejects the flags this alias sets.
Kept only for old-figure/checkpoint interpretability. Do not use for new work.

This module is the **paper-facing** entry point for the model. It is a thin
alias layer over the implementation class :class:`models.asl_t1_model.ASLT1Denoiser`,
which retains its original name so that every checkpoint state-dict key
(``aggregator.*``, ``asl_encoder.*``, ``naf_decoder.*``, ...) remains
backward-compatible.

New code, paper text, and external references should import :class:`CIGNet`
from this module:

.. code-block:: python

    from models.cig_net import CIGNet

    model = CIGNet(
        asl_hw=128, t1_hw=128, base_ch=32, depth=4,
        use_mossm_encoder=True,
        mossm_blocks_per_scale=2,
        mossm_n_directions=1,
        mossm_d_state=16,
        mossm_t1_gated=True,           # P2
        cmam_k_rank_div=4,             # P3
        use_naf_fusion=True,           # NAFNet decoder body
        use_mdta_fusion=True,          # Restormer MDTA channel attention
    )

The full architecture, training protocol, and selection criterion are
documented in ``ASL_dmvae/docs/archive/v2_mossm/cig_net.md``.

Design contracts (see paper §2 / ASL_dmvae/docs/archive/v2_mossm/cig_net.md §2):

  C1  Lesion-safe aggregation         — SetTransformerAggregator
  C2  T1 content isolation (V = ASL)  — MoSSM (P2 gated B/Δ) + CMAM (P3 low-rank K)
  C3  T1-free decoder                 — NAFDecoder with no T1 entry point
  C4  Unbiased SSL selection          — constrained-uMSE (runner-side)
"""

from __future__ import annotations

from .asl_t1_model import ASLT1Denoiser as CIGNet

# Make the alias discoverable by ``from models.cig_net import *``.
__all__ = ["CIGNet"]
