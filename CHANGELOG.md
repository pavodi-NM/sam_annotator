# Changelog

All notable changes to SAM Annotator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1a12] - 2025-12-20

### Fixed
- **Critical:** `--model_type` parameter was not correctly passed from CLI to SAM predictor, causing model architecture mismatches with checkpoint files
- Default checkpoint paths now correctly match the specified model_type
- Size mismatch errors when loading SAM1 models (vit_b, vit_l)

### Added
- Comprehensive test suite for all SAM1 model types (vit_h, vit_b, vit_l)
- Comprehensive test suite for all SAM2 model types (tiny, small, base, large, and v2 variants)
- Parametrized tests for model_type parameter propagation
- CLI parser tests for model_type handling

### Improved
- Suppressed FutureWarning from segment-anything dependency (temporary fix until upstream update)
- Version synchronization now automatic using importlib.metadata (single source of truth in pyproject.toml)
- No more manual version updates needed in `__init__.py`

### Migration
See [MIGRATION_0.2.1a12.md](MIGRATION_0.2.1a12.md) for detailed migration information.

## [0.2.1a11] - Previous Release

Earlier releases - see git history for details.
