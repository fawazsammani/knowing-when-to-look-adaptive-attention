import json
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
import torch.backends.cudnn as cudnn
from models import *
from util import *
from dataset import *

#Model Parameters
emb_dim = 512                  # dimension of word embeddings
attention_dim = 512            # attention hidden size
hidden_size = 512              # dimension of decoder RNN
cudnn.benchmark = True         # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
# Training parameters
start_epoch = 0
epochs = 40                             # number of epochs to train before finetuning the encoder. Set to 18 when finetuning ecoder
epochs_since_improvement = 0            # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 80                         # set to 32 when finetuning the encoder
workers = 1                             # number of workers for data-loading
encoder_lr = 1e-4                       # learning rate for encoder. if fine-tuning, change to 1e-5 for CNN parameters only
decoder_lr = 5e-4                       # learning rate for decoder
grad_clip = 0.1                         # clip gradients at an absolute value of
best_cider = 0.                         # Current BLEU-4 score 
print_freq = 100                        # print training/validation stats every __ batches
fine_tune_encoder = False                # set to true after 20 epochs 
checkpoint = None    # path to checkpoint, None at the begining
annFile = 'cococaptioncider/annotations/captions_val2014.json'  # Location of validation annotations


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
        enc_image,  global_features = encoder(imgs)
        predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind = decoder(enc_image, global_features, 
                                                                                         caps, caplens)
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

def validate(val_loader, encoder, decoder, beam_size, epoch, vocab_size):
    """
    Funtion to validate over the complete dataset
    """
    encoder.eval()
    decoder.eval()
    results = []

    for i, (img, image_id) in enumerate(tqdm(val_loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        infinite_pred = False
        
        # Encode
        image = img.to(device)       # (1, 3, 224, 224)
        enc_image, global_features = encoder(image) # enc_image of shape (1,num_pixels,features)
        # Flatten encoding
        num_pixels = enc_image.size(1)
        encoder_dim = enc_image.size(2)
        # We'll treat the problem as having a batch size of k
        enc_image = enc_image.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
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
        h, c = decoder.init_hidden_state(enc_image)
        spatial_image = F.relu(decoder.encoded_to_hidden(enc_image))  # (k,num_pixels,hidden_size)
        global_image = F.relu(decoder.global_features(global_features))      # (1,embed_dim)
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (k,embed_dim)
            inputs = torch.cat((embeddings, global_image.expand_as(embeddings)), dim = 1)    
            h, c, st = decoder.LSTM(inputs , (h, c))  # (batch_size_t, hidden_size)
            # Run the adaptive attention model
            out_l, _, _ = decoder.adaptive_attention(spatial_image, h, st)
            # Compute the probability over the vocabulary
            scores = decoder.fc(out_l)      # (batch_size, vocab_size)
            scores = F.log_softmax(scores, dim=1)   # (s, vocab_size)
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
            spatial_image = spatial_image[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                infinite_pred = True
                break

            step += 1
            
        if infinite_pred is not True:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = seqs[0][:20]
            seq = [seq[i].item() for i in range(len(seq))]
                
        # Construct Sentence
        sen_idx = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        sentence = ' '.join([rev_word_map[sen_idx[i]] for i in range(len(sen_idx))])
        item_dict = {"image_id": image_id.item(), "caption": sentence}
        results.append(item_dict)
    
    print("Calculating Evalaution Metric Scores......\n")
    
    resFile = 'cococaptioncider/results/captions_val2014_results_' + str(epoch) + '.json' 
    evalFile = 'cococaptioncider/results/captions_val2014_eval_' + str(epoch) + '.json' 
    # Calculate Evaluation Scores
    with open(resFile, 'w') as wr:
        json.dump(results,wr)
        
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)
    # evaluate on a subset of images
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    # evaluate results
    cocoEval.evaluate()    
    # Save Scores for all images in resFile
    with open(evalFile, 'w') as w:
        json.dump(cocoEval.eval, w)

    return cocoEval.eval['CIDEr'], cocoEval.eval['Bleu_4']


with open('caption data/WORDMAP_coco.json', 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}  # idx2word

if checkpoint is None:
    decoder = DecoderWithAttention(hidden_size = hidden_size,
                                   vocab_size = len(word_map), 
                                   att_dim = attention_dim, 
                                   embed_size = emb_dim,
                                   encoded_dim = 2048) 
    
    encoder = Encoder(hidden_size = hidden_size, embed_size = emb_dim)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr, betas = (0.8,0.999))
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr, betas = (0.8,0.999)) if fine_tune_encoder else None
    
else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_cider = checkpoint['cider']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    if fine_tune_encoder is True and encoder_optimizer is None:
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),lr=encoder_lr)
        print("Finetuning the CNN")

# Move to GPU, if available
decoder = decoder.to(device)
encoder = encoder.to(device)

# Loss function
criterion = nn.CrossEntropyLoss().to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(COCOTrainDataset(transform=transforms.Compose([normalize])),
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)

val_loader = torch.utils.data.DataLoader(COCOValidationDataset(transform=transforms.Compose([normalize])),
                                         batch_size = 1,
                                         shuffle=True, 
                                         pin_memory=True)

# Epochs
for epoch in range(start_epoch, epochs):
    
    # Terminate training if there is no improvmenet for 8 epochs
    if epochs_since_improvement == 8:
        print("No Improvement for the last 6 epochs. Training Terminated")
        break
    
    # Decay the learning rate by 0.8 every 3 epochs
    if epoch % 3 == 0 and epoch !=0:
        adjust_learning_rate(decoder_optimizer, 0.8)

    # One epoch's training
    train(train_loader=train_loader,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch,
          vocab_size = len(word_map))
    
    # One epoch's validation
    recent_cider, recent_bleu4 = validate(val_loader = val_loader, 
                                          encoder = encoder, 
                                          decoder = decoder,
                                          beam_size = 3, 
                                          epoch = epoch, 
                                          vocab_size = len(word_map))
    
    print("Epoch {}:\tCIDEr Score: {}\tBLEU-4 Score: {}".format(epoch, recent_cider, recent_bleu4))

    # Check if there was an improvement
    is_best = recent_cider > best_cider
    best_cider = max(recent_cider, best_cider)
    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
        epochs_since_improvement = 0

    save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, recent_cider, is_best)
