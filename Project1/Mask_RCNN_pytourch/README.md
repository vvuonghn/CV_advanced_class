# MaskRCNN for chair detection and segmentation in ADE20K dataset
### This is a simple implementation of a MaskRCNN for object detection and instance segmentation.

### Data preprocessing
Before training, we need to preprocessing origin dataset. 
In: processing_data.py
1) Set folder of the trainings set in data_path
2) Set folder for save output dataset after preprocessing in out_path

### Training
In: main.py

1) Set TRAIN = False
2) Set folder for save model after training in path_save
3) python main.py

### Evaluating
In: main.py

1) Set TRAIN = True
2) Set path model to load in path_model
3) python main.py
