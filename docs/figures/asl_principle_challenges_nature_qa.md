# QA: ASL principle and project challenges figure

- **Backend:** Python / Matplotlib only.
- **Final dimensions:** 183 mm x 122 mm (KBS double-column layout).
- **Primary output:** `asl_principle_challenges_nature.svg`.
- **Companion outputs:** PDF and 600-dpi PNG preview.
- **SVG editability:** 87 text nodes remain editable; mathematical expressions use Matplotlib math text where needed.
- **Raster audit:** 0 embedded raster images. The brain, vessel, frame-stack, clock, and fusion illustrations are vector patches.
- **Typography:** Arial with Helvetica/DejaVu Sans fallbacks; no missing-glyph warnings in the final export.
- **Color/accessibility:** Information is redundantly encoded by labels, borders, and arrows; red/green is not the only differentiator.
- **Scientific checks:**
  - Conventional positive-perfusion subtraction is shown as control minus label.
  - The output is explicitly normalized PWI / Delta-M, not quantitative CBF.
  - The acquisition setting is labelled 7 T, single PLD, 12 NEX.
  - Few-frame inference is labelled 2-8 NEX.
  - The 12-NEX mean is described as higher SNR but still not clean truth.
  - T1w is shown as a separate anatomical prior, and unrestricted transfer is framed as a risk rather than a guaranteed outcome.
- **Image integrity:** Conceptual vector schematic; no clinical image manipulation or quantitative source data applies.

