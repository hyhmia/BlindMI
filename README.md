# BlindMI

This is the code for "Practical Blind Membership Inference Attack via Differential Comparison". We are trying to implement an MI attack, called BlindMI.

## Requirements
+ Python3.7
+ Tensorflow 2.1.0
+ Tensorflow Datasets
+ Scikit-learn
+ tqdm
+ Numpy
+ Pandas
+ Pillow
## Code Usage
dataLoader.py is to provide the data for other modules.
### Train TargetModel:
```bash
#Train TargetModel as following:
python TargetModel.py CIFAR ResNet50
# The model weights will be saved in the following folder: weights/Target. And you could change the dataset's name and model's name.
```
