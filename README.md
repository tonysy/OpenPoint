# OpenPoint
Deep learning for point sets is hot topic and there are many works try to generate more powerful representation of the point for classifcation or semantic segmentation.

In this projoect, we try to provie a flexible and easy-to-use implementation of the existing works for point cloud. It would be helful for the future research and application. We plant to support the pioneer work: PointNet[1] and PointNet++.

# Highlights
- **PyTorch 1.0**: PointNet and PointNet++ 
- **Fast**: We support distribitured training with the support of Single-node-Multi-GPU and Multi-node-Multi-GPU

# TODO List
- [ ] Framework skeleton code
- [ ] Dataloader for ModelNet40 and ScanNet
- [ ] PointNet Model
- [ ] PointNet++ Model
- [ ] Visualization for point in 3D space
- [ ] PointCNN Model 
# Model Zoo and Checkpoints
We provide the performance of different architecture in [MODEL_ZOO](./MODEL_ZOO.md)
# Installization 
## 1. Enverionment setup
```bash
# Crate Conda virtual environment
conda env create -f environment.yml

# enter virtual env
source activate openpoint

```
## 2. Build the custom operators

# Usage
## 1. Inference

## 2. Training and Evaluation

# Reference
- PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et al. (CVPR 2017).
- PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. by Qi et al. (NIPS 2017).
- Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs. By Landrieu et al. (CVPR 2018)
# Citations
Please consider citing this project in your publication if it helps your research. The following is the a BibTex reference. And the it requires the `url` laTex package.
```
@misc{syzhang2019openpoint,
author = {Songyang, Zhang and Shipeng, Yan},
title = {{OpenPoint: Efficeint and Modular Implementation of Point Classification and Semantic Implementation in PyTorch}},
year = {2018},
howpublished = {\url{https://github.com/tonysy/openpoint}},
}
```
# Acknologement
We thanks the open-source implementation of the community and list them below:
- [Pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch.git)
- [pointnet2-tensorflow](https://github.com/charlesq34/pointnet2)
- [Superpoint-graph-pytorch](https://github.com/charlesq34/pointnet2)
