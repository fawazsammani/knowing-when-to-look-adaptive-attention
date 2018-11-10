# PyTorch Implementation of Knowing When to Look: Adaptive Attention via a Visual Sentinal for Image Captioning [Paper](https://arxiv.org/abs/1612.01887)

## Result acheived on the Flickr30k Dataset:
| Implementation  | BLEU-4 Score |
| ------------- | ------------- |
| Original Implementation | **0.251**  |
| This Implementation  | **0.260**  |

## Dataset
I'm using the Flickr30k Dataset. You may download the images from [here](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities). If you wish to use the COCO Dataset, you will need to comment out 2 lines in the code. <br/>
I'm also using Karpathy's Train/Val/Test Split. You may download it from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).<br/>
You may also use the `WORMAP.json` file in the directory if you don't wish to create it again. 

## Files
`dataset.py` Creates the Custom Dataset<br/>
`inputs.py` Preprocessing of Images<br/>
`vocab.py` Creates the `WORDMAP.json` file<br/>
`util.py` Other Functions to be used throught the code<br/>
`models.py` Defines the models<br/>
`train_eval.py` For training and evaluation<br/>
`Visualization.ipynb` To visualize the results

## Trained Model
If you dont wish to train the model from scratch, you may dowload the trained model from [here](https://drive.google.com/open?id=1H1vz-WLG8AGFziNSY_z9N3BkYFsETZkD). Trained for 15 epochs without finetuning the encoder, and 5 epochs with finetuning. <br/>

## Testing
It's very simple! Place the test image in your directory, and name it as `test.jpg`, and then run the `visualization.ipynb`jupyter notebook file to get the results. 

## Results
The results of some validation and testing images of the Flickr30k from Karpathy's Split is shown below. *Betas are shown in red*<br/> <br/>
![10350842 result](https://user-images.githubusercontent.com/30661597/48299555-3375b180-e483-11e8-83e0-2798d2a17de5.png)
![121556607 result](https://user-images.githubusercontent.com/30661597/48299556-340e4800-e483-11e8-8eed-d995e8d878e6.png)
![1499495021 result](https://user-images.githubusercontent.com/30661597/48299557-340e4800-e483-11e8-8867-3dafa5449976.png)
![2209496328 result](https://user-images.githubusercontent.com/30661597/48299558-34a6de80-e483-11e8-8b5e-fd2d211f4791.png)
![2479180530 result](https://user-images.githubusercontent.com/30661597/48299559-353f7500-e483-11e8-996b-9fa7f0017281.png)


## References
Thanks to @https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning<br/>
