# Migration Guide: v0.1.0

## 🎉 Welcome to SAM Annotator v0.1.0!

This is the **first stable release** of SAM Annotator on PyPI. This guide will help you get started or migrate from TestPyPI development versions.

---

## New Installation

If you're installing SAM Annotator for the first time:

```bash
pip install sam-annotator
```

That's it! All dependencies (including segment-anything and ultralytics) will be installed automatically.

---

## Upgrading from TestPyPI Dev Versions

If you were testing with v0.2.1.dev10, v0.2.1.dev11, or v0.2.1.dev12 from TestPyPI:

```bash
# Uninstall the TestPyPI version
pip uninstall sam-annotator

# Install the stable version from PyPI
pip install sam-annotator

# Verify the installation
sam_annotator --version
# Should output: sam_annotator 0.1.0
```

**Important:** v0.2.1.dev11 had a critical missing dependency and should not be used.

---

## Key Features in v0.1.0

### 1. Support for 1000+ Classes

The class selector now supports up to 1000+ classes with a scrollable UI:
- Fixed viewport showing 20 classes at a time
- Full keyboard navigation
- Mouse wheel scrolling
- Visual scroll indicators
- Auto-scroll to keep selected class visible

**Before:** Limited to 15 classes
**After:** Up to 1000+ classes supported

### 2. Full Keyboard Navigation

New keyboard shortcuts for class selection:
- **Arrow Up/Down**: Navigate to previous/next class
- **Page Up/Down**: Jump 5 classes at a time
- **Home/End**: Jump to first/last class
- **Mouse Wheel**: Scroll viewport (3 classes per scroll)

### 3. Dual SAM Support

Supports both SAM1 and SAM2 models:
- **SAM1**: Original Segment Anything (vit_h, vit_b, vit_l)
- **SAM2**: Ultralytics SAM2 (tiny, small, base, large, plus)

### 4. Model Type Parameter Fix

Critical fix: `--model_type` parameter now correctly propagates to the predictor:

```bash
# This now works correctly
sam_annotator --sam_version sam1 --model_type vit_b \
    --category_path my_project --classes_csv classes.csv
```

**Previously:** Model type mismatches caused size errors
**Now:** Automatic checkpoint path matching

---

## Breaking Changes

**None** - This is the first stable release, so there are no breaking changes from previous stable versions.

If you're migrating from TestPyPI dev versions, the API remains the same.

---

## Configuration

### Basic Usage

```bash
# Annotate images with SAM2
sam_annotator --category_path /path/to/project --classes_csv classes.csv

# Use SAM1 with specific model
sam_annotator --sam_version sam1 --model_type vit_b \
    --category_path /path/to/project --classes_csv classes.csv

# Get help
sam_annotator --help
```

### Classes CSV Format

Your `classes.csv` should have a `class_name` column:

```csv
class_name
person
car
dog
cat
...up to 1000 classes
```

### Project Structure

```
your_project/
├── images/           # Input images
├── masks/           # Output masks (auto-created)
├── annotations/     # Output annotations (auto-created)
└── classes.csv      # Class definitions
```

---

## Known Issues & Workarounds

### FutureWarning from segment-anything

You may see warnings about `torch.load` usage. These are temporarily suppressed at the package level and will be resolved when Meta updates the segment-anything library.

**Impact:** None - this is a warning only, functionality is not affected.

---

## Bug Fixes Included

This stable release includes fixes for:

1. **Critical:** Model type parameter propagation
   - Fixed architecture mismatches with checkpoint files
   - Fixed size mismatch errors for SAM1 models

2. **Critical UI:** Arrow key navigation
   - Arrow keys now properly select classes (not just scroll viewport)
   - Fixed arrow down key conflict with view controls

3. **UI:** Mouse click scroll offset
   - Clicks now accurately target classes in scrolled lists

4. **Performance:** Key event flooding
   - Eliminated unnecessary processing of "no key pressed" events

---

## Testing

This release includes comprehensive test coverage:
- ✅ All SAM1 model types (vit_h, vit_b, vit_l)
- ✅ All SAM2 model types (tiny, small, base, large, v2 variants)
- ✅ Model type propagation through CLI → Annotator → Predictor
- ✅ Checkpoint path resolution
- ✅ Class selector UI with keyboard/mouse navigation

---

## Additional Documentation

- **[CHANGELOG.md](CHANGELOG.md)** - Complete version history
- **[CLASS_LIMIT_CHANGES.md](CLASS_LIMIT_CHANGES.md)** - Detailed UI improvements and bug fixes
- **[SAM_PREDICTOR_IMPLEMENTATION.md](SAM_PREDICTOR_IMPLEMENTATION.md)** - SAM integration details
- **[ULTRALYTICS_SAM_SUPPORT.md](ULTRALYTICS_SAM_SUPPORT.md)** - SAM2 support documentation

---

## Getting Help

- **Documentation:** https://pavodi-nm.github.io/sam_annotator/
- **Issues:** https://github.com/pavodi-nm/sam_annotator/issues
- **Source:** https://github.com/pavodi-nm/sam_annotator

---

## What's Next?

Future releases will focus on:
- Additional export formats
- Performance optimizations
- Enhanced UI features
- Community feedback and bug fixes

Welcome to SAM Annotator! 🚀
