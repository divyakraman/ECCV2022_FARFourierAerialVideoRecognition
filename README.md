### Paper - [**FAR: Fourier Aerial Video Recognition (ECCV 2022)**](https://arxiv.org/abs/2203.10694)

Please cite our paper if you find it useful.

```
@inproceedings{kothandaraman2022far,
  title={FAR: Fourier Aerial Video Recognition},
  author={Kothandaraman, Divya and Guan, Tianrui and Wang, Xijun and Hu, Shuowen and Lin, Ming and Manocha, Dinesh},
  booktitle={European Conference on Computer Vision},
  pages={657--676},
  year={2022},
  organization={Springer}
}
```

### Code structure
#### Dataloaders <br>
dataset/dataset.py

#### Models
model/i3d_resnet.py - I3D model <br>
model/x3d.py - X3D model <br>
model/discaus1.py - I3D + FAR (Ours) <br>
model/x3d_discaus.py - X3D + FAR (Ours)

### Dependencies
PyTorch <br>
NumPy <br>
Matplotlib <br>
OpenCV <br>
SciPy

## Acknowledgements

This code is heavily borrowed from [**Benchmarking Action Recognition Models**](https://github.com/IBM/action-recognition-pytorch), and [**X3D-MultiGrid-PyTorch**](https://github.com/kkahatapitiya/X3D-Multigrid)
