# HGA-STR

This code used in Python2.7.
It's the code for the paprt [A holistic representation guided attention network for scene text recognition](https://arxiv.org/abs/1904.01375) Neurocomputing 2020.

### Install the enviroment
```bash
    pip install -r requirements.txt
```
Please convert your own dataset to **LMDB** format by create_dataset.py. (Borrowed from https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py, provided by [Baoguang Shi](https://github.com/bgshih))

There are trained model, converted [Synth90K](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) LMDB dataset by [luyang-NWPU](https://github.com/luyang-NWPU): [Here](https://pan.baidu.com/s/1mUMFu693mxAlk900E6YmuQ),  password: q86m


### Training
```bash
sh ./train.sh
```

### Testing
```bash
sh ./val.sh
```

### Recognize a image
```bash
python  pre_img.py  YOUR/MODEL/PATH  YOUR/IMAGE/PATH
```

### Citation
```
@article{yang2020holistic,
  title={A Holistic Representation Guided Attention Network for Scene Text Recognition},
  author={Yang, Lu and Wang, Peng and Li, Hui and Li, Zhen and Zhang, Yanning},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier}
}
```
### Acknowledgment
This code is based on [MORAN](https://github.com/Canjie-Luo/MORAN_v2) by [Canjie-Luo](https://github.com/Canjie-Luo). Thanks for your contribution.
