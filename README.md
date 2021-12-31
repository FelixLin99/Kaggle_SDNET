<h2 align="center"><code> ⏯ Kaggle - Structural Defects Network (SDNET) 2018 </code></h2>


<p align="center">
    <img src="https://github.com/FelixLin99/Kaggle_SDNET/raw/master/tmp/img/architecture.png" 
         width="81%">
</p>

<p align="center">"<i> Train a neural network to automatically detect structural defects with 96% Val_accu! </i>"</p>

<br>
<div align="center">
  <sub>Created by
  <a href="https://github.com/FelixLin99/">@Shuhui</a>
</div>

***
# Introduction
- An exploration towards [Structural Defects Network (SDNET) 2018](https://www.kaggle.com/aniruddhsharma/structural-defects-network-concrete-crack-images)
- Utilize data augmentation with geometric transformations using Torchvision in Python
- Train the data with Resnet34 and achieved a classification of cracked/non-cracked structures with an accuracy of 96% after parameter optimization

# Requirements
- `Python` >= 3.7.0
- `CUDA` >= 11
- `Pytorch` >= 1.7.0

# Environment
- video card:    <i>NVIDA Tesla K80 × 2</i>
- Video Memory:     <i>24G</i>
- Memory:     <i>16G</i>

# Usage
To use time-frequency module, place the contents of this folder in your PYTHONPATH environment variable.</br>
To detect single image, use [predict.py](https://github.com/FelixLin99/Kaggle_SDNET/tree/master/codes/deeplearning/predict.py) and change the `img_path`：
```python
  if __name__ == '__main__':
    img_path = "../test_pic.jpg"
    main(img_path)
```

To detect a batch image, use [batch_predict.py](https://github.com/FelixLin99/Kaggle_SDNET/tree/master/codes/deeplearning/batch_predict.py) and change the `img_path_list`：
  
```python
if __name__ == '__main__':
    img_path_list = ["../tulip.jpg", "../rose.jpg"]
    main(img_path_list)
```
Then get classified image with certainty:
<div align=center>
  <img src="https://github.com/FelixLin99/Kaggle_SDNET/raw/master/tmp/img/predict/1.jpg" height=230>
  <img src="https://github.com/FelixLin99/Kaggle_SDNET/raw/master/tmp/img/predict/3.jpg" height=230>
  <img src="https://github.com/FelixLin99/Kaggle_SDNET/raw/master/tmp/img/predict/2.jpg" height=230>
    </div>
<br>

# Transfer Learning
Thanks to the [model](https://download.pytorch.org/models/resnet34-333f7ec4.pth) provided by Pytorch.
  



    
    
    
