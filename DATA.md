## Data Prepration

We use two main classes for datasets ingestion, the first is `.data/dataset_manager.py`, 
which decodes a `dataset_config.yaml` file, and registers the datasets towards training and testing. 
The second is `./data/dataset_mapper.py`, which converts the annotations for model propagation and evaluation.

Here's an example for a data configuration:
```yaml
# ./data_configs/data_config_pretrain.yaml

ROOT: /data  # This is the root directory where the dataset directories are found, each has the dataset images 
             # and an annotations.json file

DATASETS:     # These are datasets we converted to COCO format and used for training
  - SynthText_coco
  - totaltext_train_coco

VAL_DATASETS:  # Validation is performed on these datasets
  - totaltext_test_coco

```

In each dataset directory, the model expected to find the images and an `annotations.json` in [COCO format](https://cocodataset.org/)
according to the following structure:

```json
{
  "info": {},
  "licenses": [{
      "id": 1,
      "name": "<License Name>",
      "url": "<License URL>"
    }],
  "categories": [{
      "id": 1,
      "name": "word", 
      "supercategory": "documents"
  }],
  "type": "instances",
  "images": [ 
    {
      "id": 1,
      "file_name": "img199.jpg",
      "width": 2593,
      "height": 1936,
      "date_captured": "<date captured>",
      "license": 1,
      "coco_url": "",
      "url": ""
    },
    ...
  ],
  "annotations": [
    {"id": 2544, 
      "category_id": 1, 
      "category": "word", 
      "image_id": 300, 
      "word_length": 3, 
      "text": "ROC", 
      "iscrowd": 0, 
      "area": 666.5, 
      "bbox": [47.0, 111.0, 45.0, 25.0], 
      "segmentation": [[53.0, 113.0, 71.0, 118.0, 85.0, 111.0, 92.0, 126.0, 69.0, 136.0, 47.0, 126.0]], 
      "width": 360, 
      "height": 162, 
      "angle": 1.49, 
      "orientation": 0, 
      "rotated_box": [[46.21, 113.42], [91.04, 110.62], [92.53, 134.53], [47.71, 137.33]]}]}
    ...
  ]
}
  
```

We note that we enrich the baseline COCO annotations with additional fields, including:

```yaml
    'text': 'Hello'   # A string containing the transcription of the annotation in UTF-8 format
    'text_length': 5  # The length of the text
    'angle' 5.5       # The rotated box angle measured in CCW degrees 
    'orientation' 0 / 1 / 2 / 3  # This is computed by the equation "orientation = (angle + 45) // 90 % 4"
    'rotated_box:     # (2 x 4 float list), measured in absolute image pixels and degrees, marking the coordinates of the box
```


