# Severstal Steel Defect Detection Dataset Card

## Source

- Name: Severstal: Steel Defect Detection
- Publisher: PAO Severstal through Kaggle
- Source:
  https://www.kaggle.com/c/severstal-steel-defect-detection/data
- Project task: multilabel semantic segmentation

## Data Contract

- 12,568 labeled training images.
- Image resolution: 1600 x 256 pixels.
- Four anonymized defect classes.
- Pixel masks stored as run-length encoded start-length pairs.
- One image can contain no defect, one class, or multiple classes.
- Class labels and positive pixels are strongly imbalanced.

The original competition test set does not provide public ground truth, so the
versioned train/validation/test manifests are generated from the labeled
training set.

## Class Mapping

| CSV ClassId | Project label | Output channel |
| --- | --- | --- |
| 1 | `defect_1` | 0 |
| 2 | `defect_2` | 1 |
| 3 | `defect_3` | 2 |
| 4 | `defect_4` | 3 |

This mapping is a project contract shared by data conversion, training, export,
postprocessing, and API responses.

## Split Policy

- Train: 70%.
- Validation: 15%.
- Test: 15%.
- Seed: 42.
- Strategy: iterative multilabel stratification over the four positive-class
  indicators plus the no-defect indicator.
- Leakage control: each source image appears in exactly one split.

Generated manifests contain relative paths, class indicators, source file
size, and SHA256. The test manifest is not used for threshold selection.

## Validation Gates

The raw dataset is rejected when:

- required CSV columns are absent;
- an image or annotation row is missing;
- class identifiers fall outside 1 through 4;
- an RLE sequence is malformed;
- expected image or annotation counts change;
- a split manifest contains duplicate or overlapping source images.

## Repository Policy

Dataset images and annotations are not committed to Git. The repository stores
source references, validation code, split manifests, derived aggregate
statistics, and representative outputs permitted by the source terms. Users
must obtain the data through Kaggle and accept the applicable competition
rules.

## Known Limitations

- Defect class semantics are anonymized.
- Images originate from one steel-production domain.
- Public data cannot reproduce camera, lighting, or process drift.
- Offline quality does not establish production safety or line-readiness.
- The benchmark demonstrates engineering methodology, not a certified
  inspection system.
