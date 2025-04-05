# Loading and Saving Annotations

SAM Annotator provides robust functionality for loading and saving annotations in various formats. This guide explains how to manage your annotation data effectively.

## Annotation Storage Structure

When you specify a category path for your annotations, SAM Annotator creates the following directory structure:

```
category_path/
├── images/     # Place your images here
├── labels/     # Annotation files will be saved here
├── masks/      # Visualization of masks will be saved here
├── metadata/   # Metadata about annotations
└── exports/    # Exported annotations in various formats
```

## File Formats

SAM Annotator uses a text-based format for storing annotations internally, but can export to multiple industry-standard formats. 

### Internal Format

Annotations are stored in the `labels/` directory with `.txt` files corresponding to each image. Each line represents a polygon annotation in the following format:

```
class_id x1 y1 x2 y2 ... xn yn
```

Where:
- `class_id` is the numeric ID of the annotation class
- `x1 y1, x2 y2, ...` are normalized coordinates (0-1) of the polygon vertices

### Visualization

For each annotated image, SAM Annotator creates a visualization in the `masks/` directory. This visualization shows:
- The segmentation mask on the left
- The original image with a semi-transparent overlay on the right

These visualizations make it easy to review your annotations visually.

## Automatic Saving

Annotations are automatically saved when:
1. You press the `S` key
2. You navigate to another image (using `N` or `P` keys)
3. You exit the application

## Loading Annotations

When you open an image that has existing annotations, SAM Annotator automatically loads them. The process works as follows:

1. SAM Annotator looks for a corresponding `.txt` file in the `labels/` directory
2. If found, it loads the polygon coordinates and scales them to match the current display dimensions
3. Annotations are displayed on the image with their assigned class colors

## Manual Importing (Coming Soon)

Support for manually importing annotations from other tools and formats is planned for a future release.

## Caching

SAM Annotator implements a caching mechanism to improve performance when working with large datasets. This means:

1. Loaded annotations are kept in memory for faster access
2. The cache is automatically cleared when needed to manage memory usage
3. You can manually clear the cache by restarting the application

## Common Issues

### Missing Annotations

If annotations aren't appearing for an image:
- Check if the corresponding `.txt` file exists in the `labels/` directory
- Verify that the filename matches the image (same name, different extension)
- Ensure the annotation file has the correct format

### Corrupted Annotation Files

If annotation files become corrupted:
1. Look for backup files in the `backups/` directory
2. Restore the backup by copying it to the `labels/` directory with the correct filename

## Best Practices

1. **Create Regular Backups**: Use the backup functionality to save your progress
2. **Consistent Naming**: Keep image filenames consistent to avoid issues with annotation matching
3. **Check Visualizations**: Review the mask visualizations to ensure annotations are correct
4. **Export Regularly**: Export your annotations to standard formats periodically 