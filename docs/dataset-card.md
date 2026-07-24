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
- The current Kaggle export uses a sparse annotation table: only positive
  image-class masks are listed, and absent images are normal samples.
- One image can contain no defect, one class, or multiple classes.
- Class labels and positive pixels are strongly imbalanced.

Validated aggregate statistics:

| Item | Count |
| --- | ---: |
| Annotation rows / positive masks | 7,095 |
| Images referenced by positive annotations | 6,666 |
| Normal images inferred from the image directory | 5,902 |
| `defect_1` masks | 897 |
| `defect_2` masks | 247 |
| `defect_3` masks | 5,150 |
| `defect_4` masks | 801 |

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
- Strategy: deterministic label-powerset stratification over the four
  positive-class indicators, including the all-zero normal signature.
- Leakage control: each source image appears in exactly one split.

The generated manifest contains image identifiers, split assignment, a
no-defect/defect indicator, and four class indicators. The report records the
annotation and manifest SHA256 values. The test split is not used for
threshold selection.

## Validation Gates

The raw dataset is rejected when:

- required CSV columns are absent;
- a referenced image is missing;
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
