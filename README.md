
<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="./img_src/Columbia-Lions-Logo.png" width="200" />
</a>
</div>

# COMS-6998: Applications of Deep Learning in Surgery
#### Alexander Ruthe (ayr2111) and Skyler Szot (sls2305)

## I. Project Description

Recent advances in minimally invasive surgery have yielded 
datasets of intraoperative video recordings, well
suited for deep learning applications. This project investigates
deep learning methods applied to laparoscopic cholecystectomy 
(gallbladder removal) surgery videos specifically, comparing feature
extraction architectures, characterize them in ways not explored in the published
literature, attempting to improve them with transfer learning, and 
applying human interpretable visual results based on class activation mapping. 

<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="./img_src/project_summary.png" width="600">
</a>
</div>


#### Dataset

The dataset used here is [CholecT45](https://github.com/CAMMA-public/cholect45) [1], a dataset of laparoscopic cholecystectomy surgical videos. The dataset contains 45 videos of cholecystectomy procedures collected in Strasbourg, France. The images are extracted at 1 fps from the videos and annotated with triplet information about surgical actions in the format of <instrument, verb, target>. There are 90,489 frames and 127,385 triplet instances in the dataset. An example of three frames for six different videos is shown below.

Each video is annotated with an action triplet containing at least one of each of 7 instruments, 11 verbs, and 15 tissues:

- **Instruments:** grasper, bipolar, hook, scissors, clipper, irrigator, null_instrument
- **Verbs:** grasp, retract, dissect, coagulate, clip, cut, aspirate, irrigate, pack, null_verb
- **Instruments:** gallbladder, cystic_plate, cystic_duct, cystic_artery, cystic_pedicle, blood_vessel, fluid, abdominal_wall_cavity, liver, adhesion, omentum, peritoneum, gut, specimen_bag, null_target

<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="./img_src/video_frames_example.png" width="800">
</a>
</div>

#### Thrust 1: Surgical Video Annotation

This project classifies the endoscopic surgical videos of [1] with action triplets of format (surgical
tool, surgical action, targeted tissue) listed above using a spatiotemporal deep learning architecture called TripNet [2]. 

<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="img_src/tripnet.png" width="700">
</a>
</div>

The Tripnet model is composed of a feature extraction layer that provides input features to the encoder and decoder in the subsequent architecture. This feature extractor is studied further in thrust 2 by comparing fetaure extraction models and thrust 3 by evaluating transfer learning methods. 

The TripNet model encoder encodes triplet components using a Weakly-Supervised Localization (WSL) layer that localizes the instruments. Moreover, the Class Activation Guide (CAG) detects the verbs and targets leveraging the instrument activations.

the TripNet decoder associates triplets from multi-instances, learning instrument-verb-target associations using a learning projection and for final triplet classification.

#### Thrust 2: TripNet Characterization

Characterize the performance of the Thrust 1 architecture across different
deep learning configurations based on dropout layers and associated probability, batch
normalization layers, activation functions, batch sizes, learning rates, weight
initialization, optimizers, and input data standardization techniques. The trade space will
be evaluated for convergence speed and classification accuracy.

- Weights and Balances plan
- Feature extracor model comparison plan
- Downsampling Plan

#### Thrust 3: Transfer Learning

Implement a transfer learning method using non-gallbladder tissue datasets
such as gastrointestinal dataset [3] to pretrain the TripNet spatial feature extractor and
evaluate the change in performance for gallbladder surgical videos similar to [4]. Only
the ResNet feature extractor is pre-trained, then fine-tuned using the CholecT45 dataset.

#### Thrust 4: Explainability via Class Activation Mapping

Bring explainability of machine learning model decisions to surgical annotation
deep learning by pairing Thrust 1 architecture with class activation mappings (CAM) [5].


## II. Repository Description






## III. Example Commands

```
------------------------------ Starting New Test ------------------------------
Model: Resnet18
Compute Device Assigned: Tesla V100-SXM2-16GB
Dataset Loaded: cholect45-crossval
Resnet18 Model Built
Metrics Built
Model Weights Loaded
Experiment started ...
   logging outputs to:  ./__checkpoint__/run_0/tripnet_cholectcholect45-crossval_k1_lowres.log
| resnet18 | epoch  1/10 | batch    0|
| resnet18 | epoch  1/10 | batch   10|
| resnet18 | epoch  1/10 | batch   20|
| resnet18 | epoch  1/10 | batch   30|
| resnet18 | epoch  1/10 | batch   40|
| resnet18 | epoch  1/10 | batch   50|
| resnet18 | epoch  1/10 | batch   60|
| resnet18 | epoch  1/10 | batch   70|
| resnet18 | epoch  1/10 | batch   80|
| resnet18 | epoch  1/10 | batch   90|
| resnet18 | epoch  1/10 | batch  100|
```

#### Requirements
The model depends on the following libraries:
1. sklearn
2. PIL
3. Python >= 3.5
4. ivtmetrics
5. Developer's framework:
    1. For Tensorflow version 1:
        * TF >= 1.10
    2. For Tensorflow version 2:
        * TF >= 2.1
    3. For PyTorch version:
        - Pyorch >= 1.10.1
        - TorchVision >= 0.11

<br />

#### System Requirements:
The code has been test on Linux operating system. It runs on both CPU and GPU.
Equivalence of basic OS commands such as _unzip, cd, wget_, etc. will be needed to run in Windows or Mac OS.

<br />

#### Quick Start
* clone the git repository: ``` git clone https://github.com/CAMMA-public/tripnet.git ```
* install all the required libraries according to chosen your framework.
* download the dataset
* download model's weights
* train
* evaluate

<br />


* All frames are resized to 256 x 448 during training and evaluation.
* Image data are mean normalized.
* The dataset variants are tagged in this code as follows: 
   - cholect50 = CholecT50 with split used in the original paper.
   - cholect50-challenge = CholecT50 with split used in the CholecTriplet challenge.
   - cholect45-crossval = CholecT45 with official cross-val split **(currently public released)**.
   - cholect50-crossval = CholecT50 with official cross-val split.

<br />


#### Evaluation Metrics

The *ivtmetrics* computes AP for triplet recognition. It also support the evaluation of the recognition of the triplet components.
```
pip install ivtmetrics
```
or
```
conda install -c nwoye ivtmetrics
```
Usage guide is found on [pypi.org](https://pypi.org/project/ivtmetrics/).

<br />


#### Running the Model

The code can be run in a trianing mode (`-t`) or testing mode (`-e`)  or both (`-t -e`) if you want to evaluate at the end of training :

<br />

##### Training on CholecT45/CholecT50 Dataset

Simple training on CholecT50 dataset:
```
python run.py -t  --data_dir="/path/to/dataset" --dataset_variant=cholect50 --version=1
```

You can include more details such as epoch, batch size, cross-validation and evaluation fold, weight initialization, learning rates for all subtasks, etc.:

```
python3 run.py -t -e  --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --kfold=1 --epochs=180 --batch=64 --version=2 -l 1e-2 1e-3 1e-4 --pretrain_dir='path/to/imagenet/weights'
```

All the flags can been seen in the `run.py` file.
The experimental setup of the published model is contained in the paper.

<br />

##### Testing

```
python3 run.py -e --dataset_variant=cholect45-crossval --kfold 3 --batch 32 --version=1 --test_ckpt="/path/to/model-k3/weights" --data_dir="/path/to/dataset"
```

<br />

##### Training on Custom Dataset

Adding custom datasets is quite simple, what you need to do are:
- organize your annotation files in the same format as in [CholecT45](https://github.com/CAMMA-public/cholect45) dataset. 
- final model layers can be modified to suit your task by changing the class-size (num_tool_classes, num_verb_classes, num_target_classes, num_triplet_classes) in the argparse.

<br />


## IV. Results


Dataset ||Components AP ||||| Association AP |||
:---:|:---:|:---:|:---: |:---:|:---:|:---:|:---:|:---:|:---:|
.. | AP<sub>I</sub> | AP<sub>V</sub> | AP<sub>T</sub> ||| AP<sub>IV</sub> | AP<sub>IT</sub> | AP<sub>IVT</sub> |
CholecT40 | 89.7 | 60.7 | 38.3 ||| 35.5 | 19.9 | 19.0|
CholecT45 | 89.9 | 59.9 | 37.4 ||| 31.8 | 27.1 | 24.4|
CholecT50 | 92.1 | 54.5 | 33.2 ||| 29.7 | 26.4 | 20.0|

<br />


#### PyTorch
| Network   | Base      | Resolution | Dataset   | Data split  |  Link             |
------------|-----------|------------|-----------|-------------|-------------------|
| Tripnet   | ResNet-18 | Low        | CholecT50 | RDV         |   [Google] [Baidu] |
| Tripnet   | ResNet-18 | High       | CholecT50 | RDV         |   [Google] [Baidu] |
| Tripnet   | ResNet-18 | Low        | CholecT50 | Challenge   |   [Google] [Baidu] |

<br />


# V. References

[1] A.P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy, EndoNet: A
Deep Architecture for Recognition Tasks on Laparoscopic Videos, IEEE Transactions on Medical
Imaging (TMI), arXiv preprint, 2017

[2] Nwoye, Chinedu Innocent, et al. "Recognition of instrument-tissue interactions in endoscopic
videos via action triplets." International Conference on Medical Image Computing and
Computer-Assisted Intervention. Springer, Cham, 2020.

[3] Borgli, Hanna, et al. "HyperKvasir, a comprehensive multi-class image and video dataset for
gastrointestinal endoscopy." Scientific data 7.1 (2020): 1-14.

[4] Christodoulidis, Stergios, et al. "Multisource transfer learning with convolutional neural
networks for lung pattern analysis." IEEE journal of biomedical and health informatics 21.1
(2016): 76-84.

[5] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via
gradient-based localization." Proceedings of the IEEE international conference on computer
vision. 2017.