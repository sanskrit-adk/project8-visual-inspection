# Decision Log

Each important design decision, the alternatives considered, and the evidence behind it.
Evidence paths point to files the code writes in `results/`.

| Decision | Alternatives considered | Why rejected | Evidence (file/figure) | Phase |
|---|---|---|---|---|
| One `config.yaml` for all paths and hyperparameters; seed 42 everywhere | Hard-coded constants per script; argparse-only settings | Hard-coded values drift between scripts and team members; argparse-only settings aren't recorded with the results | `config.yaml`, `results/metrics/environment.json` | 0 |
| Device auto-detect CUDA → MPS → CPU, plus a `--fast` mode (128px, 3 epochs, 25% subset) | GPU-only code; separate laptop scripts | GPU-only blocks team members without a GPU; duplicate scripts would diverge | `src/utils.py` (`get_device`, `apply_fast`) | 0 |
| Keep 3-channel RGB input with ImageNet normalisation even though castings are grey | Convert to 1-channel greyscale | Pretrained ResNet weights expect 3 channels; changing the first layer discards part of the pretrained knowledge we want to test | `config.yaml` (`image`) | 0 |
