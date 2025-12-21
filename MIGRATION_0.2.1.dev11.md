# Migration Guide: v0.2.1.dev11

## Overview

This development release includes critical bug fixes for model type handling and major UI improvements for the class selector, expanding support from 15 to 1000+ classes with full keyboard navigation.

## Bug Fixes

### Critical: Model Type Parameter Propagation

**Issue:** In versions prior to 0.2.1.dev11, the `--model_type` parameter was not correctly passed to the SAM predictor, causing model architecture mismatches with checkpoint files.

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

## Class Limit Expansion (15 → 1000+)

**Major Feature:** Class selector now supports 1000+ classes with scrollable UI.

### What Changed:
- Maximum classes increased from 15 to 1000
- UI window fixed at 20 visible classes (600px height)
- Full keyboard navigation support
- Mouse wheel scrolling support
- Auto-scroll to keep selected class visible

### Action Required:
None - existing projects will work seamlessly with the expanded limit.

### New Keyboard Shortcuts:
- **Arrow Up/Down**: Navigate to previous/next class
- **Page Up/Down**: Jump 5 classes at a time
- **Home/End**: Jump to first/last class
- **Mouse Wheel**: Scroll viewport (3 classes at a time)

For complete details, see [CLASS_LIMIT_CHANGES.md](CLASS_LIMIT_CHANGES.md)
