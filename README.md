# UGI-endoscopy
## List folders in repository:

### (1) data-exploration:
- data-exploration.ipynb: file to explore dataset and number of image labels per class
- filtered-labels: folder containing the files with filtered label names
- not-filtered: folder containing the image label data from the HyperKvasir Dataset

### (2) data-preparation:
- data-augmentation.py: file for data augmentation
- split-videos.py: file where videos were split into frames

### (3) experiment1:
- resnet50-1.ipynb: ResNet50 model for experiment 1
- vgg16-1.ipynb: VGG16 model for experiment 1

### (4) experiment2:
- vgg16-2.ipynb: VGG16 model for experiment 2

### (5) experiment3: 
- resnet50-3.ipynb: ResNet50 model for experiment 3
- vgg16-3.ipynb: VGG16 model for experiment 3

### (6) experiment4:
- vgg16-3.ipynb: VGG16 model for experiment 4

### (7) testing-data
Contains the split UCL video frames organised in folders with the class names

### (8) training-data: contains HyperKvasir Data organised in folders with the class names
- other1: this folder corresponds to the 'other' class built with pathological finding images
- other2: this folder corresponds to the 'other' class built with mucosal view images