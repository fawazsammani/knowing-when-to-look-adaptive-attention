# PyTorch Implementation of Knowing When to Look: Adaptive Attention via a Visual Sentinal for Image Captioning [Paper](https://arxiv.org/abs/1612.01887)

## Result acheived on the Flickr30k Dataset:
| Implementation  | BLEU-4 Score |
| ------------- | ------------- |
| Original Implementation | **0.2541**  |
| This Implementation  | **0.2600**  |

## Dataset
I'm using the Flickr30k Dataset. You may download the images from [here](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities). If you wish to use the COCO Dataset, you will need to comment out 2 lines in the code. <br/>
I'm also using Karpathy's Train/Val/Test Split. You may download it from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).<br/>
You may also use the `WORMAP.json` file in the directory if you don't wish to create it again. 

## Files
`dataset.py` Creates the Custom Dataset
`inputs.py` Preprocessing of Images
`vocab` Creates the `WORDMAP.json` file
`util` Other Functions to be used throught the code
`models` Defines the models
`train_eval` For training and evaluation
`Visualization.ipynb` To visualize the results

## Trained Model
If you dont wish to train the model from scratch, you may dowload the trained model from [here](https://drive.google.com/open?id=1H1vz-WLG8AGFziNSY_z9N3BkYFsETZkD). Trained for 15 epochs without finetuning the encoder, and 5 epochs with finetuning. <br/>

## Testing
It's very simple! Place the test image in your directory, and name it as `test.jpg`, and then run the `Visualization.ipynb`jupyter notebook file to get the results. 

## Results
The results of some validation and testing images of the Flickr30k from Karpathy's Split is shown below. <br/> <br/>
![res1](https://user-images.githubusercontent.com/30661597/47791821-cbcba380-dcd7-11e8-940c-2c548e908a7d.png)
![res2](https://user-images.githubusercontent.com/30661597/47791823-cbcba380-dcd7-11e8-8756-d6ebaf039ed3.png)
![res3](https://user-images.githubusercontent.com/30661597/47791824-cbcba380-dcd7-11e8-9860-ad20bbe44be8.png)

## References
Thanks to @https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning<br/>
