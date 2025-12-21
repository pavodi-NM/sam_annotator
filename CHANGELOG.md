# Changelog

All notable changes to SAM Annotator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1.dev11] - 2025-12-21

### Fixed
- **Critical:** `--model_type` parameter was not correctly passed from CLI to SAM predictor, causing model architecture mismatches with checkpoint files
- Default checkpoint paths now correctly match the specified model_type
- Size mismatch errors when loading SAM1 models (vit_b, vit_l)
- **Critical UI:** Arrow keys now properly select classes and trigger callbacks (previously only scrolled viewport without selecting)
- **Critical UI:** Arrow down key no longer intercepted by view controls 'T' toggle shortcut
- **UI:** Mouse clicks on scrolled class list now select correct class (scroll offset bug fixed)
- **UI:** Eliminated key event flooding from key code 255 (no key pressed)
- **UI:** Fixed handler priority to give class selector precedence over view controls for arrow keys

### Added
- **Major Feature:** Expanded class limit from 15 to 1000+ classes with scrollable UI
- Scrollable class selector with fixed viewport (20 visible classes at a time)
- Full keyboard navigation for class selection (Arrow Up/Down, Page Up/Down, Home/End)
- Mouse wheel scrolling support (3 classes per scroll)
- Visual scroll indicators ("^ More above", "v More below")
- Auto-scroll to keep selected class visible in viewport
- Comprehensive test suite for all SAM1 model types (vit_h, vit_b, vit_l)
- Comprehensive test suite for all SAM2 model types (tiny, small, base, large, and v2 variants)
- Parametrized tests for model_type parameter propagation
- CLI parser tests for model_type handling

### Improved
- Suppressed FutureWarning from segment-anything dependency (temporary fix until upstream update)
- Version synchronization now automatic using importlib.metadata (single source of truth in pyproject.toml)
- No more manual version updates needed in `__init__.py`
- Event handler order optimized for better keyboard input processing

### Documentation
- See [MIGRATION_0.2.1.dev11.md](MIGRATION_0.2.1.dev11.md) for detailed migration information
- See [CLASS_LIMIT_CHANGES.md](CLASS_LIMIT_CHANGES.md) for complete documentation of class limit expansion and bug fixes

## [0.2.1.dev10] - Previous TestPyPI Release

Earlier releases - see git history for details.
