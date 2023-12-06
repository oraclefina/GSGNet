# GSGNet
This project provides the code for **'Global Semantic-Guided Network for Saliency Prediction'**, Knowledge-Based Systems. [Paper link](https://www.sciencedirect.com/science/article/pii/S0950705123010274).

## Tips
✨Note that the following evaluations are based on ✨**.PNG/.JPEG** files for **image saliency datasets** containing SALICON, TORONTO and PASCAL-S.
### A slight operation in post-processing may hurt your performance on KL and IG
```python
import numpy as np
import cv2

def post_processing_A(x, img_size):
    x = cv2.resize(x, img_size)
    x = (x - x.min()) / (x.max()-x.min())
    x = x * 255
    x = np.clip(np.round(x),0,255)
    cv2.imwrite('test.png', x)

def post_processing_B(x, img_size):
    x = cv2.resize(x, img_size)
    x = (x - x.min()) / (x.max()-x.min())
    x = x * 255
    x = x + 0.5 # ✨
    x = np.clip(np.round(x),0,255)
    cv2.imwrite('test.png', x)

```

We have the following results on SALICON test set:
|   |   AUC ↑  |   CC ↑ |         KL ↓       |         IG ↑       |  sAUC ↑ |  NSS ↑  |      SIM ↑     |
|:-:|:-----:|:-----:|:-----------------:|:-----------------:|:-----:|:-----:|:-------------:|
| A | 0.869 | 0.913 |     **0.256**     |     **0.873**     | 0.746 | 1.988 |     0.808     |
| B | 0.870 | 0.912 | **0.190**(-0.066) | **0.907**(+0.034) | 0.746 | 1.988 | 0.800(-0.008) |

From the above results, we can observe that (A) obtains worse results on KL and IG. 
A slight operation (without +0.5) can hurt your KL and IG scores.
A lot of saliency models ([MSINet](https://github.com/alexanderkroner/saliency), [DINet](https://github.com/ysyscool/DINet), [TranSalNet](https://github.com/LJOVO/TranSalNet), etc.) suffer from this issue, which results in abnormal KL-IG scores on SALICON, TORONTO and PASCAL-S.
For [SAM-ResNet](https://github.com/marcellacornia/sam), the situation is tricky. 
Even though SAM-ResNet is trained with KL, you should bear in mind that SAM-ResNet uses ReLU instead of sigmoid as the final output and its implement of KL is slightly different from other saliency models'.

KL and IG seem to be sensitive to the float-integer conversion (from [0..1] to [0..255]). 


✨If you find **your model trained with a KL loss** achieving bad performance on KL and IG but obtaining great performance on other metrics, please check your post-processing method.

### Different dataset settings influence your sAUC on TORONTO and PASCAL-S
Note that the calculation of sAUC is based on the [MATLAB code](https://github.com/cvzoya/saliency/tree/master/code_forMetrics).

Basically, we have the following two types of saliency models based on the datasets used:
1. SALICON
2. SALICON + MIT1003

|  TORONTO |  AUC ↑  |   CC ↑ |         KL ↓       |         IG ↑       |  sAUC ↑ |  NSS ↑  |      SIM ↑     |
|:--------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|     1    | 0.8717 |  0.7564 | 0.5321 | 0.9236 | **0.7242** | 2.1556 | 0.6255 |
|     2    | 0.8865 | 0.8224 | 0.4341 | 1.0867 | **0.7055** |  2.4265 | 0.6738 |
|          | +0.0148 | +0.0660  | -0.0980  | +0.1631 | **-0.0187** | +0.2709 | +0.0483 |

| PASCAL-S |  AUC ↑  |   CC ↑ |         KL ↓       |         IG ↑       |  sAUC ↑ |  NSS ↑  |      SIM ↑     |
|:--------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|     1    | 0.8966 | 0.7074 | 0.7126 | 1.0762 | **0.7604** | 2.3549 | 0.5519 |
|     2    | 0.9115 | 0.7593 | 0.6077 | 1.2435 | **0.7388** | 2.6164 | 0.6074 |
|          | +0.0149 | +0.0519 | -0.1049 | +0.1673 | **-0.0216** | +0.2615 | +0.0555 |

From the above results, we can observe that the model trained on SALICON obtains better sAUC scores.  Continually finetuning on MIT1003 helps our saliency model achieve better performance on other metrics on TORONTO and PASCAL-S but worsens the sAUC scores. 
The reason might be the inherent bias within the datasets.
Since sAUC penalizes models for center bias, SALICON has a more divergent center bias, while MIT1003 has a more centralized center bias.
For a fair comparision, it is better to be evaluated under similar dataset settings.

✨If you look for a better saliency benchmarking method, please check the log probabilistic evaluation on the [new MIT benchmark](https://saliency.tuebingen.ai/).


## Requirements
```
timm==0.5.4
torch==1.9.1+cu111
```


## Testing
Clone this repository and download the pretrained weight of GSGNet trained on SALICON dataset from this [link](https://github.com/oraclefina/GSGNet/releases/tag/v1.0.0). 
Then just run the code using 
```bash
$ python inference.py --path path/to/images --weight_path path/to/model --format format/of/images
```

## Training

The dataset directory structure should be 
```
└── Dataset  
    ├── fixations
    │   ├── test
    │   ├── train
    │   └── val
    ├── images
    │   ├── test
    │   ├── train
    │   └── val
    └── maps
        ├── train
        └── val
```
Set up dataset path in config.py and run the following command to train 

```bash
$ python train.py 
```

## Citation
If you think this project is helpful, please feel free to cite our paper:

    @article{Xie2023gsgnet,
    title = {Global semantic-guided network for saliency prediction},
    journal = {Knowledge-Based Systems},
    pages = {111279},
    year = {2023},
    issn = {0950-7051},
    doi = {https://doi.org/10.1016/j.knosys.2023.111279},
    author = {Jiawei Xie and Zhi Liu and Gongyang Li and Xiaofeng Lu and Tao Chen},
    }
        
