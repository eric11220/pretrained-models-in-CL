## Official repository of *Do Pre-trained Models Benefit Equally in Continual Learning?*
----
This repository is adapted from [Online Continual Learning](https://github.com/RaptorMai/online-continual-learning).


## Requirements
Create a virtual enviroment
```sh
conda create -n online_cl
```
Activating a virtual environment
```sh
conda activate online_cl
```
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


## Algorithms 
* EWC++: Efficient and online version of Elastic Weight Consolidation(EWC) (**ECCV, 2018**) [[Paper]](http://arxiv-export-lb.library.cornell.edu/abs/1801.10112)
* iCaRL: Incremental Classifier and Representation Learning (**CVPR, 2017**) [[Paper]](https://arxiv.org/abs/1611.07725)
* LwF: Learning without forgetting (**ECCV, 2016**) [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37)
* AGEM: Averaged Gradient Episodic Memory (**ICLR, 2019**) [[Paper]](https://openreview.net/forum?id=Hkf2_sC5FX)
* ER: Experience Replay (**ICML Workshop, 2019**) [[Paper]](https://arxiv.org/abs/1902.10486)
* MIR: Maximally Interfered Retrieval (**NeurIPS, 2019**) [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
* GSS: Gradient-Based Sample Selection (**NeurIPS, 2019**) [[Paper]](https://arxiv.org/pdf/1903.08671.pdf)
* GDumb: Greedy Sampler and Dumb Learner (**ECCV, 2020**) [[Paper]](https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf)
* SCR: Supervised Contrastive Replay (**CVPR Workshop, 2021**) [[Paper]](https://arxiv.org/abs/2103.13885) 


## Duplicate results
- Please see scripts/run_all.sh for ways to run different models/datasets/CL algortihms.
```sh
cd scripts; bash run_all.sh pretrained_rn18 cifar100 20 0
```

## Citation 

If you use this paper/code in your research, please consider citing us:

**Do Pre-trained Models Benefit Equally in Continual Learning?**

```
```


## Contact & Contribution
- [Kuan-Ying Lee](https://zheda-mai.github.io/) (Corresponding author)  
kylee5@illinois.edu
- [Yuanyi Zhong]
yuanyiz2@illinois.edu
- [Yuxiong Wang]
yxw@illinois.edu


## Acknowledgments
- [Online Continual Learning](https://github.com/RaptorMai/online-continual-learning)
