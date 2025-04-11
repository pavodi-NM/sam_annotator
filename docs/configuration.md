# Configuration Options

SAM Annotator provides various configuration options to customize its behavior. This guide explains the available options, how to set them, and their effects on the annotation process.

## Command Line Arguments

When running SAM Annotator, you can provide several command line arguments to configure its behavior:

```bash
sam_annotator --category_path <path> --classes_csv <path> [--sam_version sam1|sam2] [--model_type <type>] [--checkpoint <path>]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--category_path` | Path to the root directory for your annotation project |
| `--classes_csv` | Path to the CSV file containing class definitions |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--sam_version` | SAM version to use ('sam1' or 'sam2') | 'sam1' |
| `--model_type` | Model type for the selected SAM version | 'vit_h' (SAM1) or 'small_v2' (SAM2) |
| `--checkpoint` | Path to a custom SAM model checkpoint | Default weights for selected model |

## Class Definition File

The class definition file is a CSV file that defines the annotation classes available in your project. The file should have the following format:

```csv
class_id,class_name,color
0,background,"0,0,0"
1,person,"255,0,0"
2,car,"0,255,0"
...
```

### Fields

- `class_id`: Numeric ID for the class (starts at 0)
- `class_name`: Display name for the class
- `color`: RGB color values for the class visualization (comma-separated)

!!! note
    SAM Annotator currently supports a maximum of 15 classes (including background).

## Configuration Files (Coming Soon)

Support for configuration files to customize additional aspects of SAM Annotator is planned for a future release. This will include:

- UI customization options
- Default visualization settings
- Performance optimization settings
- Custom keyboard shortcuts

## Environment Variables

SAM Annotator supports the following environment variables for advanced configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `SAM_CACHE_DIR` | Directory for caching model weights | `~/.cache/sam_annotator` |
| `SAM_LOG_LEVEL` | Logging verbosity level | `INFO` |
| `SAM_DEVICE` | Device to run inference on | `cuda` if available, otherwise `cpu` |

## Model Configuration

### SAM Version 1

When using SAM version 1 (`--sam_version sam1`), the following model types are available:

| Model Type | Description | Size | Memory |
|------------|-------------|------|--------|
| `vit_h` | ViT-Huge backbone | ~2.4GB | ~16GB VRAM |
| `vit_l` | ViT-Large backbone | ~1.2GB | ~8GB VRAM |
| `vit_b` | ViT-Base backbone | ~375MB | ~4GB VRAM |

### SAM Version 2

When using SAM version 2 (`--sam_version sam2`), the following model types are available:

| Model Type | Description | Size | Memory |
|------------|-------------|------|--------|
| `tiny` | Super-lightweight model | ~36MB | ~2GB VRAM |
| `small` | Small model | ~47MB | ~3GB VRAM |
| `base` | Base model | ~93MB | ~5GB VRAM |
| `large` | Large model | ~312MB | ~8GB VRAM |
| `tiny_v2` | Enhanced tiny model | ~46MB | ~3GB VRAM |
| `small_v2` | Enhanced small model | ~86MB | ~4GB VRAM |
| `base_v2` | Enhanced base model | ~166MB | ~6GB VRAM |
| `large_v2` | Enhanced large model | ~637MB | ~10GB VRAM |

## View Configuration

SAM Annotator allows you to customize the visualization of annotations during the annotation process:

| Setting | Keyboard Shortcut | Description |
|---------|------------------|-------------|
| Mask Visibility | `M` | Show/hide segmentation masks |
| Box Visibility | `B` | Show/hide bounding boxes |
| Label Visibility | `L` | Show/hide class labels |
| Point Visibility | `T` | Show/hide prompt points |
| Mask Opacity | `[` / `]` | Decrease/increase mask opacity |

## Performance Considerations

When configuring SAM Annotator, consider the following performance trade-offs:

1. **Model Selection**: Larger models (vit_h, large_v2) provide better segmentation quality but require more VRAM and are slower.
2. **Image Size**: Larger images will consume more memory and slow down processing.
3. **Batch Processing**: SAM Annotator processes images one at a time, so CPU/GPU utilization may not be optimal for all configurations.

## Best Practices

1. **Start Small**: Begin with a smaller model if you're unsure about your hardware capabilities.
2. **Test Performance**: Try different configurations on a small subset of your data before starting large annotation projects.
3. **Consider Image Resolution**: Downscaling very high-resolution images before annotation can improve performance.
4. **Monitor Memory Usage**: Watch for out-of-memory errors, especially when using larger models.
5. **Save Frequently**: Remember to save annotations frequently regardless of your configuration. 