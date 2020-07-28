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
+ OpenCV
## Code Usage
dataLoader.py is to provide the data for other modules.

### Train TargetModel:
```bash
#Train TargetModel as following:
python TargetModel.py CIFAR ResNet50
```
The model weights will be saved in the following folder: weights/Target. And you could change the dataset's name and model's name, which is included in dataLoader.py and ModelUtil.py seperately.

### BlindMI-Diff Attack:
+ BlindMI-Diff-w Attack
```bash
#Try BlindMI-Diff-w as following:
python BlindMI_Diff_W.py CIFAR ResNet50
```
This attack is based on a non-Member gernerated function and only use 20 generated non-Members to extraly prode the target model.
You could use the other Datasets and Target Models trained in last section. The evaluation results will be output directly as it finishes.

You can also try the following attack and the difference is that it will use the whole gernerated non-Members .

```bash
#Try BlindMI-Diff-w as following:
python BlindMI_Diff_W_Ori.py CIFAR ResNet50
```

+ BlindMI-Diff-w/o:
```bash
#Try BlindMI-Diff-w/o as following:
python BlindMI_Diff_Without_Gen.py CIFAR ResNet50
```
This is under the condition that we cannot use non-Member gernerated by ourselves. It has two ways to get a rough part of non-member data.





