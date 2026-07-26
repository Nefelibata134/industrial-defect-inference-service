# ONNX Export and Parity

## Purpose

This report verifies that the promoted PyTorch checkpoint can be converted to
an ONNX graph without changing its numerical or thresholded segmentation
behavior. It also freezes the model interface used by TensorRT and Triton.

The regression set is designed to detect conversion defects. Full validation
metrics remain the source of truth for model quality.

## Artifact Identity

| Artifact | SHA256 |
| --- | --- |
| PyTorch checkpoint | `a6a2a76857ba0a1a857a78204f77a82f58c09f26996f41ba51bac42479393ccc` |
| ONNX graph | `6622d3aada8f7fe0547c475f80a8697cb74f7e6e8c6b677a9dbaff45a4f03a0c` |

The checkpoint contains the U-Net ResNet-18 parameters selected at epoch 2.
The ONNX graph is exported with ONNX IR version 10 and opset 18.

## Model Contract

| Field | Contract |
| --- | --- |
| Input name | `images` |
| Input type | float32 |
| Input shape | `[batch, 3, 256, 1024]` |
| Output name | `logits` |
| Output type | float32 |
| Output shape | `[batch, 4, 256, 1024]` |
| Dynamic dimension | Batch only |
| Class order | `defect_1`, `defect_2`, `defect_3`, `defect_4` |
| Postprocessing | Sigmoid, then threshold `0.80` |

The graph returns logits. Keeping sigmoid and thresholding outside the graph
allows deployment clients to apply the same calibrated policy while retaining
access to continuous scores.

## Protocol

1. Rebuild the U-Net ResNet-18 architecture and strictly load the promoted
   checkpoint parameters.
2. Switch the model to evaluation mode and export with a float32 input of
   shape `[1, 3, 256, 1024]`.
3. Validate the graph with the ONNX checker and assert the expected names,
   shapes, types, opset, and metadata.
4. Load the same validation images into PyTorch and ONNX Runtime CPU.
5. Compare raw logits, thresholded masks, and macro Dice.
6. Repeat inference with a different batch size to exercise the dynamic
   dimension without re-exporting the graph.

## Acceptance Criteria

| Check | Limit |
| --- | ---: |
| Maximum absolute logit error | `<= 1e-3` |
| Mean absolute logit error | `<= 1e-5` |
| Thresholded-mask mismatch ratio | `<= 1e-6` |
| Macro-Dice absolute delta | `<= 1e-6` |

## Results

Environment:

- PyTorch `2.12.1+cu130`
- ONNX Runtime `1.23.2`
- ONNX Runtime provider: CPU
- Regression samples: 8 validation images

| Metric | Result |
| --- | ---: |
| Maximum absolute logit error | `9.72747802734375e-05` |
| Mean absolute logit error | `1.9139116034239123e-06` |
| Thresholded-mask mismatched pixels | `0` |
| Thresholded-mask mismatch ratio | `0.0` |
| Macro-Dice absolute delta | `0.0` |
| Status | **Pass** |

Batch sizes `2` and `4` produced the same passing result. No re-export is
required when only the symbolic batch dimension changes.

## Reproduction

```bash
python scripts/export_onnx.py
python scripts/inspect_onnx.py
python scripts/verify_onnx_parity.py \
  --samples 8 \
  --batch-size 2 \
  --report outputs/reports/onnx_parity.json
```

The ONNX graph is a generated release artifact and is excluded from source
control. Its source identity and verification evidence are retained through
the checkpoint hash, ONNX hash, versioned report, configuration, tests, and
Git revision.
