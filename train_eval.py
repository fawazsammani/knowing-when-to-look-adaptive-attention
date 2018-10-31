import json
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import *
from models import *
from util import *

#Get what you have, CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model Parameters
emb_dim = 512                  # dimension of word embeddings
attention_dim = 64             # dimension of attention linear layers (k), whoch is = num_pixels, i.e 8x8
hidden_size = 512              # dimension of decoder RNN
cudnn.benchmark = True         # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
# Training parameters
start_epoch = 0
epochs = 20                             # number of epochs to train before finetuning the encoder. Set to 18 when finetuning ecoder
epochs_since_improvement = 0            # keeps track of number of epochs since there's been an improvement in validation BLEU
# if using multiple GPUs, then actual batch size = gpu_num * batch_size. One GPU is set to 20
batch_size = 30                         # set to 30 when finetuning the encoder
workers = 1                             # number of workers for data-loading
encoder_lr = 1e-5                       # learning rate for encoder if fine-tuning
decoder_lr = 4e-4                       # learning rate for decoder
grad_clip = 0.1                         # clip gradients at an absolute value of
best_bleu4 = 0.                         # Current BLEU-4 score 
print_freq = 100                        # print training/validation stats every __ batches
fine_tune_encoder = True                # set to true after 20 epochs 
checkpoint = 'BEST_checkpoint_17.pth.tar'    # path to checkpoint, None at the begining


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, vocab_size):

    decoder.train()                 # train mode (dropout and batchnorm is used)
    encoder.train()
    losses = AverageMeter()         # loss (per decoded word)
    top5accs = AverageMeter()       # top5 accuracy

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        
        # Forward prop.
        spatial_image, global_image, enc_image = encoder(imgs)
        predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind = decoder(spatial_image, global_image, 
                                                                                         caps, caplens, enc_image)
        
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = encoded_captions[:, 1:]
        
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        
        # Calculate loss
        loss = criterion(scores, targets)
        
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
#         clip_gradient(decoder_optimizer, grad_clip)
#         if encoder_optimizer is not None:
#             clip_gradient(encoder_optimizer, grad_clip)
                
        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
            
        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))    
        top5accs.update(top5, sum(decode_lengths))

        # Print status every print_freq iterations --> (print_freq * batch_size) images
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                            loss=losses,
                                                                            top5=top5accs))

def beam_search_eval(encoder, decoder, img, beam_size, wrong):
    
    k = beam_size
    vocab_size = len(word_map)
    smth_wrong = False

    # Encode
    image = img.cuda().unsqueeze(0)  # (1, 3, 256, 256)
    spatial_image, global_image, encoder_out = encoder(image) #enc_image of shape (batch_size,num_pixels,features)
    # Flatten encoding
    num_pixels = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_scores = list()
    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  
        inputs = torch.cat((embeddings,global_image.expand_as(embeddings)), dim = 1)    
        h, c = decoder.LSTM(inputs, (h, c))  # (1, hidden_size)
        # Run the sentinal model
        st = decoder.sentinal(inputs, h, c)
        alpha, ct, zt = decoder.spatial_attention(spatial_image,h)
        # Run the adaptive attention model
        c_hat, beta_t, alpha_hat = decoder.adaptive_attention(h, st, zt, ct)
        # Compute the probability
        scores = decoder.fc(c_hat + h) 
        scores = F.log_softmax(scores, dim=1)   # (s, vocab_size)
        # Add
        # (k,1) will be (k,vocab_size), then (k,vocab_size) + (s,vocab_size) --> (s, vocab_size)
        scores = top_k_scores.expand_as(scores) + scores  
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            #Remember: torch.topk returns the top k scores in the first argument, and their respective indices in the second argument
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s) 
        next_word_inds = top_k_words % vocab_size  # (s) 
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        if k == 0:
            break

        # Proceed with incomplete sequences
        seqs = seqs[incomplete_inds]              
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            smth_wrong = True
            break

        step += 1

    if smth_wrong is not True:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        sentence = [rev_word_map[seq[i]] for i in range(len(seq))]
    else:
        seq = seqs[0][:20]
        sentence = [rev_word_map[seq[i].item()] for i in range(len(seq))]
        sentence = sentence + ['<end>']
        wrong += 1
    
    return sentence,wrong

def evaluate_batch(encoder, decoder, imgs, allcaps):
    """
    Returns references and predictions using beam search for one batch, for bleu 4 calculation 
    """
    # Predict the captions of one complete batch
    wrong = 0
    batch_predictions = []
    for i in range(imgs.shape[0]):
        sent,wrong = beam_search_eval(encoder, decoder, imgs[i].cuda(),3, wrong)
        batch_predictions.append(sent)
    # Append the references
    batch_references = []
    for batch in allcaps:
        five_captions = []
        for i in range(5):
            current = batch[i] 
            current_sen = [rev_word_map[current[j].item()] for j in range(current.shape[0])]
            filtered_sen = [word for word in current_sen if word != '<pad>']
            five_captions.append(filtered_sen)
        batch_references.append(five_captions)
    return batch_predictions, batch_references, wrong

def evaluate(encoder, decoder):
    """
    Prints the loss using Teacher Forcing, and returns the bleu-4 validation on the complete dataset without using teacher
    forcing, at a beam size of 3
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()
    wrong_p = 0
    print("--------------------Evaluating---------------------")    
    all_predictions = []
    all_references = []
    for iteration, (imgs, caps, caplens, allcaps) in enumerate (val_loader):
        #imgs    --> shape (batch_size, 3, 256, 256)
        #caps    --> shape (batch_size,max_length)
        #caplens --> shape (batch_size,1)
        #allcaps --> shape (batch_size,5,max_length)
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        # Calculate the loss using Teacher Forcing
        spatial_image, global_image, enc_image = encoder(imgs)
        predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind = decoder(spatial_image, global_image, 
                                                                                         caps, caplens, enc_image)
        targets = encoded_captions[:, 1:]
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        # Calculate loss
        loss = criterion(scores, targets)
        # Evaluate using beam search with k = 3
        batch_predictions, batch_references, wrong = evaluate_batch(encoder, decoder, imgs, allcaps)
        wrong_p += wrong
        assert len(batch_references) == len(batch_predictions)
        # Calculate the bleu-4 score on the current batch
        #bleu4_batch = corpus_bleu(references_e, predictions_e)
        all_predictions.extend(batch_predictions)
        all_references.extend(batch_references)
        if iteration % 15 == 0:
            print("Iteration {}\tLoss With Teacher Forcing: {:.3f}\t{} Wrong Predictions till now".format(iteration,loss,wrong_p))

    assert len(all_references) == len(all_predictions)
    val_score = corpus_bleu(all_references, all_predictions)
    return val_score


# In[ ]:


with open('WORDMAP.json', 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}  # idx2word

if checkpoint is None:
    decoder = DecoderWithAttention(hidden_size = hidden_size, 
                                   vocab_size = len(word_map), 
                                   att_dim = attention_dim, 
                                   embed_size = emb_dim) 
    
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr, betas = (0.8,0.999))
    encoder = Encoder(hidden_size = hidden_size, embed_size = emb_dim)
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr, betas = (0.8,0.999)) if fine_tune_encoder else None
else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    if fine_tune_encoder is True and encoder_optimizer is None:
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr, betas = (0.8,0.999))
# Move to GPU, if available
decoder = decoder.to(device)
encoder = encoder.to(device)

# Loss function
criterion = nn.CrossEntropyLoss().to(device)

# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(CaptionDataset('TRAIN', transform=transforms.Compose([normalize])),
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(CaptionDataset('VAL', transform=transforms.Compose([normalize])),
                                         batch_size=batch_size, 
                                         shuffle=True)
# Epochs
for epoch in range(start_epoch, epochs):
    
    if epoch>20:
        adjust_learning_rate(decoder_optimizer, epoch)
        if fine_tune_encoder:
            adjust_learning_rate(encoder_optimizer, epoch)

    # Early Stopping if the validation score does not imporive for 6 consecutive epochs
    if epochs_since_improvement == 6:
        break

    # One epoch's training
    train(train_loader=train_loader,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch,
          vocab_size = len(word_map))
    
    # Evaluate with beam search 
    recent_bleu4 = evaluate(encoder, decoder)
    print('\n BLEU-4 Score on the Complete Dataset @ beam size of 3 - {}\n'.format(recent_bleu4))
         
    # Check if there was an improvement
    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)
    if not is_best:
        epochs_since_improvement += 1   
        print("\nEpochs since last improvement: {}\n".format(epochs_since_improvement))
    else:
        epochs_since_improvement = 0    #Reset to zero if there is any improvement 

    # Save checkpoint
    print("Finished one epoch. Saving to drive")
    save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, recent_bleu4, is_best)

