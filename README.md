## PyTorch Implementation of Knowing When to Look: Adaptive Attention via a Visual Sentinal for Image Captioning [Paper](https://arxiv.org/abs/1612.01887)<br/>
## Original Torch Implementation by Lu. et al can be found [here](https://github.com/jiasenlu/AdaptiveAttention)

## Dataset
I'm using the Flickr30k Dataset. You may download the images from [here](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities). If you wish to use the COCO Dataset, you will need to comment out 2 lines in the code. <br/>
I'm also using Karpathy's Train/Val/Test Split. You may download it from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).<br/>
You may also use the `WORMAP.json` file in the directory if you don't wish to create it again. 

## Files
`preprocess.py` Creates the `WORDMAP.json` file and the `.h5` files <br/>
`dataset.py` Creates the custom dataset<br/>
`util.py` Functions to be used throught the code<br/>
`models.py` Defines the architectures<br/> 
`train_eval` For Training and Evaluation<br/> 
`visualization.ipynb` For Testing and Visualization<br/>

## Testing
It's very simple! Place the test image in your directory, and name it as `test.jpg`, and then run the `visualization.ipynb`jupyter notebook file to get the results. 

## Results
The results of some validation and testing images of the Flickr30k from Karpathy's Split is shown below.<br/> <br/>
![final1](https://user-images.githubusercontent.com/30661597/48822223-10fc5780-ed11-11e8-9002-812ad7039f36.png)
![final2](https://user-images.githubusercontent.com/30661597/48824160-4bb5be00-ed18-11e8-9811-549d82e27427.png)

## References
Thanks to @https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning<br/>
