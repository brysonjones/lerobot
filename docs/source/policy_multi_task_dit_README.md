# Multitask DiT Policy

## Partially freezing the vision encoder

Over 90% of this policy's training step is the CLIP ViT-B/16 tower, and most of that is its
backward. `--policy.vision_encoder_trainable_layers=N` freezes the patch embeddings and every
vision layer below the last N, leaving the forward pass unchanged.

On one A100 80GB with LIBERO in bf16, `N=4` (28M trainable parameters instead of 86M):

| setting                             | samples/s | peak memory |
| ----------------------------------- | --------- | ----------- |
| default, whole tower trainable      | 206       | 62 GB       |
| `vision_encoder_trainable_layers=4` | 240       | 26 GB       |

The memory is the more useful half: frozen layers store no activations for the backward, so a batch
that previously needed gradient accumulation to fit may run in one pass.

This changes what the model learns, so treat it as a recipe choice rather than a free speedup. It
is off by default.

## Citation

If you use this work, please cite the following works:

```bibtex
@misc{jones2025multitaskditpolicy,
  author = {Bryson Jones},
  title = {Dissecting and Open-Sourcing Multitask Diffusion Transformer Policy},
  year = {2025},
  url = {https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-diffusion-transformer-policy},
  note = {Blog post}
}
```

```bibtex
@misc{trilbmteam2025carefulexaminationlargebehaviormodels,
  author       = {TRI LBM Team},
  title        = {A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation},
  year         = {2025},
  eprint       = {arXiv:2507.05331},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url          = {https://arxiv.org/abs/2507.05331}
}
```

```bibtex
@misc{bostondynamics2025largebehaviormodelsatlas,
  author       = {Boston Dynamics and TRI Research Team},
  title        = {Large Behavior Models and Atlas Find New Footing},
  year         = {2025},
  url          = {https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/},
  note         = {Blog post}
}
```
