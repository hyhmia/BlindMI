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
This attack is based on a non-Member gernerated function and only use 20 generated non-Members to prode the target model additionally.
You could use the other Datasets and Target Models trained in last section. The evaluation results will be output directly as it finishes.

You can also try the following attack and the difference is that it will use the whole generated non-Members .

```bash
#Try BlindMI-Diff-w as following:
python BlindMI_Diff_W_Ori.py CIFAR ResNet50
```

+ BlindMI-Diff-w/o:

```bash
#Try BlindMI-Diff-w/o as following:
python BlindMI_Diff_Without_Gen.py CIFAR ResNet50
```
This is used in the condition where we cannot use non-Member gernerated by ourselves. It has two ways to get a rough part of non-member data.


### BlindMI-1class:

+ One-Class SVM:

```bash
#Try BlindMI-Diff-w as following:
python BlindMI_1class.py CIFAR ResNet50
```

This attack is to use our gernerated non-Members to train a one-class classifier(you can also try another one), and applies it to the original data.

### Prior Attacks without Shadow Model:

+ Label-only Attack:

```bash
#Try BlindMI-Diff-w as following:
python Label_Only_Attack.py CIFAR ResNet50
```

+ Top1_Threshold Attack:

```bash
#Try BlindMI-Diff-w as following:
python Label_Only_Attack.py CIFAR ResNet50
```

### Prior Attacks with Shadow Model:


+ Train Shadow Model:

There are two conditions, BlackBox setting and GrayBox setting, for Shadow Model. You could train the shadow model with different model.The first parameter is DATA_NAME, the second one is Target Model and the last one is Shadow Model.

```bash
#Train ShadowModel as following:
python ShadowModel.py CIFAR VGG16
```

+ Loss_Threshold Attack:

```bash
#Try Loss_Threshold Attack as following:
python Loss_Threshold_Attack.py CIFAR ResNet50 VGG16
```

+ NN_Attack:

```bash
#Try NN_Attack Attack as following:
python NN_Attack.py CIFAR ResNet50 VGG16
```
+ Top3_NN_Attack:

```bash
#Try Top3_NN_Attack as following:
python Top3_NN_Attack.py CIFAR ResNet50 VGG16
```
+ Top2 + True label Attack:

```bash
#Try Top2 + True label Attack as following:
python Top2_With_True_NN_Attack.py CIFAR ResNet50 VGG16
```

### Defenses:
We evaluate our BlindMI under four defenses:

+ Memguard: [paper](https://arxiv.org/pdf/1909.10594.pdf) - [code](https://github.com/jjy1994/MemGuard)
+ Diiferential Privacy: [code](https://github.com/tensorflow/privacy)
+ Adverserial Regularization: [paper](https://arxiv.org/pdf/1807.05852.pdf) - [code](https://github.com/NNToan-apcs/python-DP-DL)
+ MMD+Mix-up: [paper](https://arxiv.org/pdf/2002.12062.pdf)

# Citation
Please cite the following paper if you use this code:
'''
@inproceedings{BlindMI,
  author = {Hui, Bo and Yang, Yuchen and Yuan, Haolin and Burlina, Philippe and Gong, Neil Zhenqiang and Cao, Yinzhi},
  title = {Practical Blind Membership Inference Attack via Differential Comparisons},
  booktitle = {Proceedings of the Network and Distributed System Security Symposium (NDSS'21)},
  year = {2021}, 
  month = {February},
}
'''
