# Changelog

All notable changes to SAM Annotator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-21

### 🎉 First Stable Release

This is the first stable release of SAM Annotator on PyPI! After extensive testing on TestPyPI (dev10-dev12), we're proud to release version 0.1.0 with major improvements and critical bug fixes.

### Added
- **Major Feature:** Expanded class limit from 15 to 1000+ classes with scrollable UI
  - Scrollable class selector with fixed viewport (20 visible classes at a time)
  - Full keyboard navigation (Arrow Up/Down, Page Up/Down, Home/End)
  - Mouse wheel scrolling support (3 classes per scroll)
  - Visual scroll indicators ("^ More above", "v More below")
  - Auto-scroll to keep selected class visible in viewport
- Comprehensive test suite for all SAM1 model types (vit_h, vit_b, vit_l)
- Comprehensive test suite for all SAM2 model types (tiny, small, base, large, and v2 variants)
- Parametrized tests for model_type parameter propagation
- CLI parser tests for model_type handling
- Support for both SAM1 (segment-anything) and SAM2 (ultralytics) models
- Automatic version synchronization using importlib.metadata

### Fixed
- **Critical:** `--model_type` parameter now correctly passed from CLI to SAM predictor
  - Fixes model architecture mismatches with checkpoint files
  - Fixes size mismatch errors when loading SAM1 models (vit_b, vit_l)
  - Default checkpoint paths now correctly match the specified model_type
- **Critical UI:** Arrow keys now properly select classes and trigger callbacks
  - Previously only scrolled viewport without actually selecting classes
  - Users can now navigate to all 1000+ classes via keyboard
- **Critical UI:** Arrow down key no longer intercepted by view controls 'T' toggle shortcut
  - Fixed handler priority to give class selector precedence
  - Arrow keys now work consistently for class navigation
- **UI:** Mouse clicks on scrolled class list now select correct class
  - Fixed scroll offset calculation bug
  - Clicks now accurately target the intended class regardless of scroll position
- **UI:** Eliminated key event flooding from key code 255 (no key pressed)
  - Improved performance and reduced unnecessary processing
- **Dependencies:** All required dependencies properly declared (segment-anything, ultralytics)

### Improved
- Event handler order optimized for better keyboard input processing
- Suppressed FutureWarning from segment-anything dependency (temporary fix until upstream update)
- Version management simplified - single source of truth in pyproject.toml
- No manual version updates needed in `__init__.py`

### Documentation
- Comprehensive [MIGRATION_0.1.0.md](MIGRATION_0.1.0.md) guide
- Detailed [CLASS_LIMIT_CHANGES.md](CLASS_LIMIT_CHANGES.md) documenting UI improvements and bug fixes
- [SAM_PREDICTOR_IMPLEMENTATION.md](SAM_PREDICTOR_IMPLEMENTATION.md) for SAM integration details
- [ULTRALYTICS_SAM_SUPPORT.md](ULTRALYTICS_SAM_SUPPORT.md) for SAM2 support

### Breaking Changes
None - this is the first stable release

### Installation

```bash
pip install sam-annotator
```

### Upgrade from TestPyPI dev versions

If you were testing with dev10, dev11, or dev12 from TestPyPI:

```bash
pip uninstall sam-annotator
pip install sam-annotator
```

---

## Development Releases (TestPyPI Only)

The following versions were released to TestPyPI for testing purposes only.

### [0.2.1.dev12] - 2025-12-21 [TestPyPI]
- Fixed: Re-added missing `segment-anything>=1.0` dependency

### [0.2.1.dev11] - 2025-12-21 [YANKED - Missing dependency]
- All features from 0.1.0 but missing critical dependency

### [0.2.1.dev10] - 2025-12-20 [TestPyPI]
- Early testing version
