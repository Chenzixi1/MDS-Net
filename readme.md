# MDS-Net: Multi-scale Deep Spatial Information Fusion Enhanced 3D Object Detection Network (CVPR 2020)

Zixi Chen, Wenbin Zou, Shishun Tian 

![image](https://github.com/Chenzixi1/MDS-Net/blob/master/demo.gif)

## Introduction

Our framework is implemented and tested with Ubuntu 16.04, CUDA 8.0/9.0, Python 3, Pytorch 0.4/1.0/1.1, NVIDIA Tesla V100/TITANX GPU. 

If you find our work useful in your research please consider citing our paper:



## Requirements

- **Cuda & Cudnn & Python & Pytorch**

    This project is tested with CUDA 8.0/9.0, Python 3, Pytorch 0.4/1.0/1.1, NVIDIA Tesla V100/TITANX GPU. And almost all the packages we use are covered by Anaconda.

    Please install proper CUDA and CUDNN version, and then install Anaconda3 and Pytorch.

- **My settings**

  ```shell
  source ~/anaconda3/bin/activate (python 3.6.5)
	(base)  pip list
	torch                              1.1.0
	torchfile                          0.1.0
	torchvision                        0.3.0
	numpy                              1.14.3
	numpydoc                           0.8.0
	numba                              0.38.0
	visdom                             0.1.8.9
	opencv-python                      4.1.0.25
	easydict                           1.9
	Shapely                            1.6.4.post2
  ```


## Data preparation

Download and unzip the full [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) detection dataset to the folder **/path/to/kitti/**. Then place a softlink (or the actual data) in **data/kitti/**. There are two widely used training/validation set splits for the KITTI dataset. Here we only show the setting of **split1**, you can set **split2** accordingly.

  ```shell
cd D4LCN
ln -s /path/to/kitti data/kitti
ln -s /path/to/kitti/testing data/kitti_split1/testing
  ```

Our method uses [DORN](https://github.com/hufu6371/DORN) (or other monocular depth models) to extract depth maps for all images. You can download and unzip the depth maps extracted by DORN [here](https://drive.google.com/open?id=1lSJpQ8GUCxRNtWxo0lduYAbWkkXQa2cb) and put them (or softlink) to the folder **data/kitti/depth_2/**. (You can also change the path in the scripts **setup_depth.py**)

Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage.


  ```shell
python data/kitti_split1/setup_split.py
python data/kitti_split1/setup_depth.py
  ```

Next, build the KITTI devkit eval for split1.

```shell
sh data/kitti_split1/devkit/cpp/build.sh
```

Lastly, build the nms modules

```shell
cd lib/nms
make
```

## Training

We use [visdom](https://github.com/facebookresearch/visdom) for visualization and graphs. Optionally, start the server by command line

```
sh visdom.sh
```
The port can be customized in config files. The training monitor can be viewed at [http://localhost:9891](http://localhost:9891). 

You can change the batch_size according to the number of GPUs, default: 1 GPUs with batch_size = 2.

See the configurations in **scripts/config/yolof_config** and **scripts/train.py** for details. 

``` 
sh train.sh
```

## Testing

We provide the [weights, model and config file](https://pan.baidu.com/s/1GuqU3B5ArAsTORyn9MquBQ?pwd=uhwp#list/path=%2F) on the val1 data split available to download.

Testing requires paths to the configuration file and model weights, exposed variables near the top **scripts/test.py**. To test a configuration and model, simply update the variables and run the test file as below. 

```
sh test.sh
```

## Acknowlegment
The code is heavily borrowed from [D4LCN](https://github.com/dingmyu/D4LCN) and thanks for their contribution.


## Contact

For questions regarding MDS-Net, feel free to post here or directly contact the authors (zixichen1@163.com).
