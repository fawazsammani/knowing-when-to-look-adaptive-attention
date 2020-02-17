## Adaptive Attention in PyTorch

PyTorch Implementation of *Knowing When to Look: Adaptive Attention via a Visual Sentinal for Image Captioning* [Paper](https://arxiv.org/abs/1612.01887)<br/>
 Original Torch Implementation by Lu. et al can be found [here](https://github.com/jiasenlu/AdaptiveAttention)
 
 <img width="802" alt="ss" src="https://user-images.githubusercontent.com/30661597/62519932-15e14700-b85f-11e9-88e8-9f3a36993723.png">

## Instructions
1. Download the COCO 2014 dataset from [here](http://cocodataset.org/#download). In particualr, you'll need the 2014 Training, Validation and Testing images, as well as the 2014 Train/Val annotations.

2. Download Karpathy's Train/Val/Test Split. You may download it from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).<br/>

3. If you want to do evaluation on COCO, make sure to download the COCO API from [here](https://github.com/cocodataset/cocoapi) if your on Linux or from [here](https://github.com/philferriere/cocoapi) if your on Windows. Then download the COCO caption toolkit from [here](https://github.com/tylin/coco-caption) and re-name the folder to `cococaption`. (This also requires java. Simply dowload it from [here](https://www.java.com/en/download/) if you don't have it). 

## Files
`preprocess.py` Creates the `WORDMAP.json` file and the `.h5` files <br/>
`dataset.py` Creates the custom dataset<br/>
`util.py` Functions to be used throught the code<br/>
`models.py` Defines the architectures<br/> 
`train_eval` For Training and Evaluation<br/> 
`run.ipynb` For Testing and Visualization<br/>
The folder `caption data` includes data used along with images, mainly for evaluation purposes.

## Testing
Place the test image in the folder 'test_imgs', and name it as `test.jpg`, and then run the `run.ipynb`jupyter notebook file to get the results. 

## Results
The file [here](https://drive.google.com/open?id=1TYZ2qfZz7CvsR3UTPw66FIHKmdwcQ6H5) contains the obtained evalaution scores. 
The pretrained model is provided [here](https://drive.google.com/open?id=1ThXtsi_Z0NVHnuyIB0eYqHbrLoqyyaPw) as well (trained for 12 epochs). 
Some results from Karpathy's Split is shown below. The visual grounding probability (1-beta) is shown in green. <br/> <br/>
![demo1](https://user-images.githubusercontent.com/30661597/58374547-b1f2f600-7f72-11e9-83be-3b16b190d2a9.PNG)
<br/>
<br/>
![demo2](https://user-images.githubusercontent.com/30661597/58374548-b1f2f600-7f72-11e9-9853-82869d746e31.PNG)


## References
Code adopted from [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) implementation of "Show, Attend and Tell"<br/>
