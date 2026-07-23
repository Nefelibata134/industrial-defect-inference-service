# ADR 0001: Use Severstal for Supervised Defect Segmentation

- Status: Accepted
- Date: 2026-07-23

## Context

The primary dataset must support a credible industrial inspection scenario,
pixel-level outputs, reproducible evaluation, and GPU deployment benchmarks.

Three candidates were evaluated:

| Dataset | Scale | Annotation | Decision |
| --- | ---: | --- | --- |
| NEU-DET | 1,800 images at 200 x 200 | Six-class boxes | Rejected as primary; too small and low resolution |
| MVTec AD | More than 5,000 high-resolution images | Anomaly masks | Not selected; changes the core problem to one-class anomaly detection |
| Severstal | 12,568 labeled images at 1600 x 256 | Four-class RLE masks | Selected |

## Decision

Use the labeled Severstal training set for multilabel semantic segmentation.
Generate deterministic 70/15/15 manifests using iterative multilabel
stratification. Treat no-defect images as an explicit stratification signal.

Use a four-channel sigmoid output so each class remains independently
representable. The initial model is U-Net with a ResNet-18 encoder and a
combined BCE and Dice loss.

## Consequences

- Pixel masks support defect area and shape, not only bounding boxes.
- The long strip geometry requires an explicit resize-versus-tiling decision.
- Sparse masks and class imbalance require per-class recall and false-negative
  reporting.
- The dataset cannot be redistributed; reproducibility relies on source
  instructions, validation, manifests, and hashes.
- Model training becomes more expensive than NEU-DET but remains practical on
  a single RTX-class GPU or rented GPU instance.
