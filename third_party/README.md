# Third-party dependencies

[FrontisAI/OpenRSI](https://github.com/FrontisAI/OpenRSI) is the upstream source for
Frontis-MA1 / OpenMLE-Evo. The confirmation is pinned to OpenRSI commit
`ece6cbdf115ed72c3b62643a836504d77365e3a0`; the baseline also used MLE-Bench commit
`507f92e1138bb6e40dac5c6ee7a6758e6424bf97`.

Upstream source and model weights are not redistributed. The
`openrsi_patches/` directory contains the research instrumentation needed for exact
live-population checkpoints, RNG and LLM sampling provenance, and final-submit
controls. Apply only the documented cumulative patch for reproduction.
