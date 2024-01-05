## *Do Pre-trained Models Benefit Equally in Continual Learning?*
----

This repository is an official implementation of *Do Pre-trained Models Benefit Equally in Continual Learning?* in Pytorch (WACV 2023)
[[Paper](https://openaccess.thecvf.com/content/WACV2023/html/Lee_Do_Pre-Trained_Models_Benefit_Equally_in_Continual_Learning_WACV_2023_paper.html)][[Project Website](https://kylee5.web.illinois.edu/publication/WACV23/index.html)]


## Abstract
Existing work on continual learning (CL) is primarily devoted to developing algorithms for models trained from scratch. Despite their encouraging performance on contrived benchmarks, these algorithms show dramatic performance drop in real-world scenarios. Therefore, this paper advocates the systematic introduction of pre-training to CL, which is a general recipe for transferring knowledge to downstream tasks but is substantially missing in the CL community. Our investigation reveals the multifaceted complexity of exploiting pre-trained models for CL, along three different axes: pre-trained models, CL algorithms, and CL scenarios. Perhaps most intriguingly, improvements in CL algorithms from pre-training are very inconsistent -- an underperforming algorithm could become competitive and even state of the art, when all algorithms start from a pre-trained model. This indicates that the current paradigm, where all CL methods are compared in from-scratch training, is not well reflective of the true CL objective and desired progress. In addition, we make several other important observations, including that 1) CL algorithms that exert less regularization benefit more from a pre-trained model; and 2) a stronger pre-trained model such as CLIP does not guarantee a better improvement. Based on these findings, we introduce a simple yet effective baseline that employs minimum regularization and leverages the more beneficial pre-trained model, coupled with a two-stage training pipeline. We recommend including this strong baseline in the future development of CL algorithms, due to its demonstrated state-of-the-art performance.
### Contribution
- We show the necessity of pre-trained models on more complex CL datasets and the dramatic difference in their benefits on different CL algorithms, which may overturn the comparison results between algorithms. Therefore, we suggest the community consider pre-trained models when developing and evaluating new CL algorithms.
- We show that replay-based CL algorithms seem to benefit more from a pre-trained model, compared with regularization-based counterparts.
- We propose a simple yet strong baseline based on ER and ImageNet RN50, which achieves state-of-the-art performance for CL with pre-training.


## Requirement
Installing packages
```sh
pip install -r requirements.txt
```

## Datasets 
- Split CIFAR100
- Split CUB200
- Split Mini-ImageNet

  
### Data preparation
- CIFAR10 & CIFAR100 & CUB200 will be downloaded during the first run. Note that, CUB200 would take a while for the first run.
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it in datasets/mini_imagenet/


## Duplicate results
- Please see scripts/run_all.sh for ways to run different models/datasets/CL algortihms.
```sh
cd scripts; bash run_all.sh pretrained_rn18 1000 cifar100 20 0
```


## Citation 
If you use this paper/code in your research, please consider citing us:

**Do Pre-trained Models Benefit Equally in Continual Learning?**

```
@inproceedings{lee2023pretrained_models_cl,
  author={Lee, Kuan-Ying and Zhong, Yuanyi and Wang, Yu-Xiong},
  title={Do Pre-trained Models Benefit Equally in Continual Learning?},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2023},
}
```


## Acknowledgments
This repository is built upon [Online Continual Learning](https://github.com/RaptorMai/online-continual-learning).
