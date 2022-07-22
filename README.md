# DAANet: Dual Attention Aggregating Network for Salient Object Detection

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript: 

> Y. Li, H. Wang, S. Wang, S. Dev. “DAANet: Dual Attention Aggregating Network for Salient Object Detection”, Image and Vision Computing, under review

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.

### Executive summary
Convolutional neural networks have been introduced for salient object detection (SOD) for several years which have been proven to have the ability to achieve better performance than traditional methods. In the early stage of the development of CNN in the SOD task, most of the convolutional neural networks use simple structured feature extraction methods with fully-connected layers to generate salient masks which failed to capture and aggregate sensitive information at the different down-sample stages. The feature pyramid network (FPN) based structure with encoder and decoder is more popular for semantic segmentation and salient object detection tasks, the FPN structure with attention modules can efficiently capture the important area of input feature and achieve better performance. In this paper, to improve the overall performance of salient object detection tasks, we propose a dual attention aggregating network (DAANet), which is an FPN-based deep convolutional neural network with a dual attention aggregation module (DAAM) and boundary-joint training. The DAAM considers the salient map prediction from the low-level output as pseudo-attention which can efficiently aggregate multi-scale information.  The Convolution block attention module (CBAM) in DAAM can refine the aggregation of pseudo-attention which enables better performance.  We evaluate our proposed DAANet on six benchmark datasets and analyze the effectiveness of each module of DAANet which shows that DAANet outweighs other previous approaches in many aspects. In addition, the lightweight configuration of the model can achieve an MAE of 0.051 on the DUTS-TE dataset with only 15.8 MB of parameters.

### Code Organization
All codes are written in `python`. 

### Code 
The script to reproduce all the figures, tables in the paper are as follows:
+ `main.py`: ...

### Data
+ `DUTS-TR`: ...
+ `DUTS-TE`: ...
+ `SOD`: ...
...

### Results 

+ `samples of qualitative results -  DAANet with VGG16 backbone and boundary-joint training_0/1.pdf`: It evaluates DAANet qualitatively by comparing the generated salient map and boundary with ground truth. The figure shows the effectiveness of both the DAAM module and boundary-joint training strategy.
+ `qualitative results with ResNet50 backbone and without BJD.pdf`: It compares DAANet (ResNet50 backbone) with seven different approaches, including BASNet, PiCANet, BMPM, R3Net+, PAGRN, SRM and DGRL. The results show that DAANet significantly improves the quality of the generated salient map, while DAANet can capture more details and have fewer wrong predictions than others.
+ `boundary ground-truth_0/1.pdf`: These two figures show four samples of using the Prewitt operator to implement a simple boundary extraction function to extract the accurate boundary from provided saliency map ground-truth.
+ `DUTS-OMRON_pr_curves.pdf`:
+ `DUTS-TE_pr_curves.pdf`:
+ `DUT-OMRON_fm_curves.pdf`:
+ `DUTS-TE_fm_curves.pdf`:
+ `ECSSD_fm_curves.pdf`:
+ `ECSSD_pr_curves.pdf`:
+ `HKU-IS_fm_curves.pdf`:
+ `HKU-IS_pr_curves.pdf`:
+ `PASCAL-S_fm_curves.pdf`:
+ `PASCAL-S_pr_curves.pdf`:
