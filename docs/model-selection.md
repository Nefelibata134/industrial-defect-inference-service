# Model Selection Report

## Scope

This report selects the segmentation checkpoint that will be exported to ONNX
and used for runtime parity and deployment benchmarks. Model selection uses the
1,885-image validation split only. The 1,885-image test split remains frozen.

All candidates use a four-channel U-Net with a ResNet-18 encoder at
`1024x256`. Macro Dice is computed from dataset-level pixel totals. Thresholds
are selected on the validation split.

## Candidate Comparison

| Candidate | Epochs | Best threshold | Macro Dice | Decision |
| --- | ---: | ---: | ---: | --- |
| Random sampling baseline | 5 | 0.05 | 0.2851 | Rejected |
| Class-aware sampling, power 0.50 | 5 | 0.70 | 0.3846 | Rejected |
| Class-aware sampling, power 0.75 | 5 | 0.80 | **0.3998** | **Promoted** |
| Controlled sampling with augmentation | 15 | 0.50 | 0.3206 | Rejected |
| Auxiliary classification with soft gating | 15 | 0.50 | 0.3292 | Rejected |

The extended candidates improved the common classes but collapsed classes 1
and 2. They were rejected because their added sampling and graph complexity did
not improve the validation objective.

## Promoted Checkpoint

- Architecture: single-head U-Net, ResNet-18 encoder
- Loss: BCE with logits plus soft Dice
- Sampler: class-aware, sampling power `0.75`
- Selected epoch: `2`
- Validation threshold: `0.80`
- Checkpoint SHA-256:
  `a6a2a76857ba0a1a857a78204f77a82f58c09f26996f41ba51bac42479393ccc`

| Metric | Class 1 | Class 2 | Class 3 | Class 4 | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dice | 0.2258 | 0.3449 | 0.5484 | 0.4799 | 0.3998 |
| IoU | 0.1273 | 0.2084 | 0.3778 | 0.3157 | 0.2573 |
| Precision | 0.2677 | 0.3082 | 0.6421 | 0.4166 | 0.4087 |
| Recall | 0.1952 | 0.3914 | 0.4786 | 0.5660 | 0.4078 |

The rare second class is learned rather than suppressed: its validation Dice is
`0.3449`, with two image-level false negatives at the selected threshold.

## Reproduction

Run threshold selection:

```bash
python scripts/evaluate_thresholds.py \
  --device cuda \
  --checkpoint models/class_aware_p075_e05_best_unet_resnet18.pt
```

Generate the aggregate and per-image error report:

```bash
python scripts/analyze_errors.py \
  --device cuda \
  --checkpoint models/class_aware_p075_e05_best_unet_resnet18.pt \
  --threshold 0.80 \
  --focus-class 2
```

Render representative best-overlap, false-positive, false-negative, and
mislocalized cases:

```bash
python scripts/visualize_errors.py \
  --device cuda \
  --analysis-report outputs/reports/error_analysis.json
```

Generated checkpoints, CSV files, JSON reports, and comparison panels stay
outside source control. This versioned report records the promoted artifact
identity and the metrics required for downstream parity checks.
