# Migration Guide: v0.2.1a12

## Bug Fixes

### Critical: Model Type Parameter Propagation

**Issue:** In versions prior to 0.2.1a12, the `--model_type` parameter was not correctly passed to the SAM predictor, causing model architecture mismatches with checkpoint files.

**Symptoms:**
- Size mismatch errors when loading SAM1 models (vit_b, vit_l)
- RuntimeError: "Error(s) in loading state_dict for Sam"
- Dimension mismatches (e.g., expecting 1280 but got 768)

**Fix:**
- Model type is now correctly propagated from CLI to SAMAnnotator to predictor
- Default checkpoint paths now match the specified model type

**Action Required:** None - existing usage will work correctly.

**Example:**
```bash
# This now works correctly (previously failed)
sam_annotator --sam_version sam1 --model_type vit_b \
    --category_path my_project --classes_csv classes.csv
```

### Warning Suppression

**Issue:** FutureWarning from segment-anything's torch.load usage

**Fix:** Temporarily suppressed at package initialization level. This is a dependency issue that will be resolved when Meta updates segment-anything.

## Testing

This release includes comprehensive tests for:
- All SAM1 model types (vit_h, vit_b, vit_l)
- All SAM2 model types (tiny, small, base, large, and v2 variants)
- Model type propagation through CLI → Annotator → Predictor
- Checkpoint path resolution

## Version Synchronization

The version is now automatically synchronized from pyproject.toml:
- No more manual version updates needed in `__init__.py`
- Single source of truth in `pyproject.toml`
- Works for both installed and development/editable installs
