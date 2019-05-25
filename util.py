import numpy as np
import torch
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import math


def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])     
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(0.001)  # pause a bit so that plots are updated


#Functon to clip gradients computed during backpropagation to avoid exploding gradient problem
#Parameters: optimizer used, and grad_clip = clip value
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

#Function to save the model checkpoint
def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, cider, is_best):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'cider': cider,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    
    filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)

#Function to shrink the learning rate by a specified factor in the interval (0,1) to multiply the learning rate with
def adjust_learning_rate(optimizer, shrink_factor):

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

#Function to compute the top-k accuracy, from predicted and true labels.
def accuracy(scores, targets, k):

    batch_size = targets.size(0)
    #Return the indices of the top-k elements along the first dimension (along every row of a 2D Tensor), sorted
    _, ind = scores.topk(k, 1, True, True)    
    #The target tensor is the same for each of the top-k predictions (words). Therefore, we need to expand it to  
    #the same shape as the tensor (ind)
    #(double every label in the row --> so every row will contain k elements/k columns) 
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))   
    #Sum up the correct predictions --> we will now have one value (the sum)
    correct_total = correct.view(-1).float().sum() 
    #Devide by the batch_size and return the percentage 
    return correct_total.item() * (100.0 / batch_size)

#Class that keeps track of most recent value, average, sum, and count of a metric.
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
