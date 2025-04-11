# Annotation Formats

SAM Annotator supports multiple annotation formats for exporting your segmentation data. This guide explains the different formats, their structure, and how to use them.

## Supported Export Formats

SAM Annotator currently supports exporting annotations to the following formats:

1. **COCO**: The Common Objects in Context format
2. **YOLO**: You Only Look Once format
3. **Pascal VOC**: Pascal Visual Object Classes format

## Exporting Annotations

You can export your annotations using the following keyboard shortcuts:

| Format      | Shortcut    |
|-------------|-------------|
| COCO        | `E`         |
| YOLO        | `Y`         |
| Pascal VOC  | `V`         |

All exports are saved to the `exports/` directory within your category path, with a subdirectory for each format.

## Format Details

### COCO Format

[COCO](https://cocodataset.org/) is a large-scale object detection, segmentation, and captioning dataset format. SAM Annotator exports annotations in the COCO format as a JSON file with the following structure:

```json
{
  "info": { ... },
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1000,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    },
    ...
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "none"
    },
    ...
  ]
}
```

#### Use Cases:
- Compatible with COCO API and many deep learning frameworks
- Ideal for object detection and instance segmentation tasks
- Good for large datasets with multiple classes

### YOLO Format

[YOLO](https://github.com/ultralytics/yolov5) is a popular real-time object detection system. SAM Annotator exports segmentation annotations in the YOLO segmentation format:

```
# One file per image
# Each line: class_id x1 y1 x2 y2 ... xn yn
# Where coordinates are normalized (0-1)
0 0.507 0.425 0.531 0.421 0.529 0.437 0.506 0.441
1 0.322 0.661 0.359 0.654 0.356 0.674 0.320 0.681
...
```

This format is compatible with YOLOv5, YOLOv8 and other versions that support segmentation.

#### Use Cases:
- Real-time object detection and segmentation
- Deployment on edge devices
- Training custom YOLO models

### Pascal VOC Format

[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) is a dataset for object class recognition. SAM Annotator exports annotations in Pascal VOC XML format:

```xml
<annotation>
  <folder>images</folder>
  <filename>image1.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>200</xmax>
      <ymax>250</ymax>
    </bndbox>
    <polygon>
      <point>
        <x>120</x>
        <y>160</y>
      </point>
      <point>
        <x>180</x>
        <y>160</y>
      </point>
      ...
    </polygon>
  </object>
  ...
</annotation>
```

#### Use Cases:
- Compatible with older computer vision tools
- Standard format for many object detection datasets
- Easy to parse and inspect manually

## Choosing the Right Format

The best format to export your annotations depends on your specific needs:

- **COCO**: Choose for machine learning frameworks that support COCO format, such as TensorFlow Object Detection API, Detectron2, or MMDetection
- **YOLO**: Best for YOLO-based models, particularly if you're using Ultralytics' YOLO implementations
- **Pascal VOC**: Good for compatibility with older systems or when XML-based formats are required

## Format Conversion (Coming Soon)

Support for converting between different annotation formats directly within SAM Annotator is planned for a future release.

## External Tools

For more advanced conversions or formats not yet supported by SAM Annotator, consider these external tools:

1. [Roboflow](https://roboflow.com/) - Supports multiple format conversions and dataset preprocessing
2. [LabelMe](https://github.com/wkentaro/labelme) - Another annotation tool with format conversion utilities
3. [COCO-to-YOLO](https://github.com/ultralytics/JSON2YOLO) - Scripts for converting COCO JSON to YOLO format

## Best Practices

1. **Export Regularly**: Export your annotations frequently to prevent data loss
2. **Version Control**: Keep track of different versions of your exported annotations
3. **Metadata**: Document any preprocessing or specific details about your exports
4. **Validation**: Verify that exported annotations are correctly loaded in your target application 