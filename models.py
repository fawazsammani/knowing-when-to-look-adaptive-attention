import torch
import torch.nn as nn
import torchvision
import torch.optim
import torch.utils.data
from torch.nn import init
import torch.nn.functional as F

#Get what you have, CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super(Encoder,self).__init__()
        #resnet = torchvision.models.resnet101(pretrained = True)
        resnet = torchvision.models.resnet101(pretrained = True)
        all_modules = list(resnet.children())
        #Remove the last FC layer used for classification and the average pooling layer
        modules = all_modules[:-2]
        #Initialize the modified resnet as the class variable
        self.resnet = nn.Sequential(*modules) 
        self.spatial_features = nn.Linear(2048, hidden_size)
        self.global_features = nn.Linear(2048, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()
        self.fine_tune()    # To fine-tune the CNN, self.fine_tune(status = True)
    
    def init_weights(self):
        """
        Initialize the weights of the spatial and global features, since we are applying a transformation
        """
        init.kaiming_uniform_(self.spatial_features.weight)
        init.kaiming_uniform_(self.global_features.weight)
        self.spatial_features.bias.data.fill_(0)
        self.global_features.bias.data.fill_(0)
    
    def forward(self,images):
        """
        The forward propagation function
        input: resized image of shape (batch_size,3,224,224)
        """
        #Run the image through the ResNet
        encoded_image = self.resnet(images)                                     # (batch_size,2048,7,7)
        batch_size = encoded_image.shape[0]
        features = encoded_image.shape[1]
        num_pixels = encoded_image.shape[2] * encoded_image.shape[3]
        #Get the global image features: ag = 1/k[sum(ai)]
        global_f = encoded_image.view(batch_size,features,num_pixels)           # (batch_size,features,num_pixels)
        global_f = global_f.mean(dim=2)                                         # (batch_size, features)
        #Reshape the encoded image to get it ready for transformation
        enc_image = encoded_image.view(batch_size,num_pixels,features)          # (batch_size,num_pixels,features)
        #Get the spatial representation
        spatial_image = F.relu(self.spatial_features(self.dropout(enc_image)))  # (batch_size,num_pixels,hidden_size)
        # Get the global features 
        global_image = F.relu(self.global_features(self.dropout(global_f)))     # (batch_size,embed_size)
        return spatial_image, global_image, enc_image
    
    def fine_tune(self, status = False):
        """
        Allows fine-tuning of the encoder after some epochs
        """
        if not status:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            for module in list(self.resnet.children())[7:]:    #1 layer only. len(list(resnet.children())) = 8
                for param in module.parameters():
                    param.requires_grad = True 

class Sentinal(nn.Module):
    def __init__(self,embed_input,hidden_size):
        super(Sentinal, self).__init__()
        self.hidd_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.inp_gate = nn.Linear(embed_input, hidden_size, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()
    
    def init_weights(self):
        init.xavier_uniform_(self.hidd_gate.weight)
        init.xavier_uniform_(self.inp_gate.weight)
    
    def forward(self,concat_input, decoder_hidden, memory_cell):
        """
        concat_input: concatenation of the word embedding vector and the global image features retuend by the encoder. The word
        embedding at a single timestep, of shape: (batch_size, embed_size) and global_image of shape (batch_size, embed_size)
        Therefore, concat_input of shape: (batch_size, embed * 2)
        decoder_hidden: hidden state of the decoder, of shape: (batch_size, hidden_size)
        memory_cell: memory state of the decoder, of shape: (batch_size, hidden_size)
        """
        sen_hidd = self.hidd_gate(self.dropout(decoder_hidden))     # (batch_size, hidden_size)
        sen_input = self.inp_gate(self.dropout(concat_input))       # (batch_size, hidden_size)
        gt = F.sigmoid(sen_hidd + sen_input)                        # (batch_size, hidden_size)
        st = gt * F.tanh(memory_cell)                               # (batch_size, hidden_size)
        return st

class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, att_dim):
        super(AdaptiveAttention,self).__init__()
        # We will set the attention dimension to the same number of pixels, i.e. 49
        self.cnn_att = nn.Linear(hidden_size, att_dim, bias=False)
        self.dec_att = nn.Linear(hidden_size, att_dim, bias=False)
        self.sen_out = nn.Linear(hidden_size,att_dim, bias=False)
        self.att_out = nn.Linear(att_dim,1, bias=False)
        self.softmax = nn.Softmax(dim=1)   
        self.dropout = nn.Dropout(0.5)
        self.init_weights()
        
    def init_weights(self):
        init.xavier_uniform_(self.cnn_att.weight)
        init.xavier_uniform_(self.dec_att.weight)
        init.xavier_uniform_(self.sen_out.weight)
        init.xavier_uniform_(self.att_out.weight)
    
    def forward(self,spatial_image, decoder_out, st):
        """
        spatial_image: the spatial image features retuend by the encoder of size (batch_size,num_pixels,hidden_size)
        decoder_out: the decoder hidden state of shape (batch_size, hidden_size)
        st: visual sentinal returned by the Sentinal class, of shape: (batch_size, hidden_size)
        """
        cnn_out = self.cnn_att(self.dropout(spatial_image))     # (batch_size, num_pixels, att_dim)
        dec_out = self.dec_att(self.dropout(decoder_out))       # (batch_size, att_dim)
        addition_out = F.tanh(cnn_out + dec_out.unsqueeze(1))   # (batch_size, num_pixels, att_dim)
        zt = self.att_out(self.dropout(addition_out)).squeeze(2)     # (batch_size, num_pixels)
        alpha_t = self.softmax(zt)                              # (batch_size, num_pixels)
        # (batch_size,num_pixels,hidden_size) * (batch_size, num_pixels, 1) = (batch_size, num_pixels, hidden_size)
        ct = (spatial_image * alpha_t.unsqueeze(2)).sum(dim=1)  # (batch_size, hidden_size)        
        out = F.tanh(self.dec_att(self.dropout(decoder_out)) + self.sen_out(self.dropout(st)))     # (batch_size,att_dim)
        out = self.att_out(self.dropout(out))                              # (batch_size, 1)
        concat = torch.cat((zt,out), dim=1)                                # (batch_size, num_pixels+1)
        alpha_hat = self.softmax(concat)                                   # (batch_size, num_pixels+1)
        beta_t = alpha_hat[:,-1].unsqueeze(1)                              # (batch_size,1)
        c_hat = beta_t * st + (1 - beta_t) * ct                            # (batch_size, hidden_size)
        return alpha_t, beta_t, c_hat


class DecoderWithAttention(nn.Module):
    def __init__(self,hidden_size, vocab_size, att_dim, embed_size):
        super(DecoderWithAttention,self).__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.sentinal = Sentinal(embed_size * 2,hidden_size)
        self.adaptive_attention = AdaptiveAttention(hidden_size, att_dim)
        # input to the LSTMCell should be of shape (batch, input_size). Remember we are concatenating the word with
        # the global image features, therefore out input features should be embed_size * 2
        self.LSTM = nn.LSTMCell(embed_size * 2, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)  
        self.init_h = nn.Linear(2048, hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(2048, hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()
        
    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, enc_image):

        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param enc_image: encoded images, a tensor of dimension  (batch_size,num_pixels,2048)
        :return: hidden state, cell state, initialized with the mean of the pixels of the encoded image
        """
        #Find the mean of all the pixels (columns of the batch) for every batch
        mean_enc_image = enc_image.mean(dim=1)           # (batch_size,2048)
        h = self.init_h(mean_enc_image)                  # (batch_size, hidden_size)
        c = self.init_c(mean_enc_image)                  # (batch_size, hidden_size)
        return h, c
    
    def forward(self, spatial_image, global_image, encoded_captions, caption_lengths, enc_image):
        
        """
        spatial_image: the spatial image features returned by the Encoder, of shape: (batch_size,num_pixels,hidden_size)
        global_image: the global image features returned by the Encoder, of shape: (batch_size, embed_size)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        enc_image: the encoded images from the encoder, of shape (batch_size, num_pixels, 2048)
        """
        batch_size = spatial_image.shape[0]
        num_pixels = spatial_image.shape[1]
        # Sort input data by decreasing lengths
        # caption_lenghts will contain the sorted lengths, and sort_ind contains the sorted elements indices 
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        #The sort_ind contains elements of the batch index of the tensor encoder_out. For example, if sort_ind is [3,2,0],
        #then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0. 
        spatial_image = spatial_image[sort_ind]           # (batch_size,num_pixels,hidden_size) with sorted batches
        global_image = global_image[sort_ind]             # (batch_size, embed_size) with sorted batches
        encoded_captions = encoded_captions[sort_ind]     # (batch_size, max_caption_length) with sorted batches 
        enc_image = enc_image[sort_ind]                   # (batch_size, num_pixels, 2048)

        # Embedding. Each batch contains a caption. All batches have the same number of rows (words), since we previously
        # padded the ones shorter than max_caption_length, as well as the same number of columns (embed_dim)
        embeddings = self.embedding(encoded_captions)     # (batch_size, max_caption_length, embed_dim)

        # Initialize the LSTM state
        h,c = self.init_hidden_state(enc_image)          # (batch_size, hidden_size)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores,alphas and betas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        betas = torch.zeros(batch_size, max(decode_lengths),1).to(device) 
        
        # Concatenate the embeddings and global image features for input to LSTM 
        global_image = global_image.unsqueeze(1).expand_as(embeddings)
        inputs = torch.cat((embeddings,global_image), dim = 2)    # (batch_size, max_caption_length, embed_dim * 2)

        #Start decoding
        for timestep in range(max(decode_lengths)):
            # Create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep. Note
            # that we cannot use the pack_padded_seq provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > timestep for l in decode_lengths])
            current_input = inputs[:batch_size_t, timestep, :]             # (batch_size_t, embed_dim * 2)
            # First, keep a copy of the hidden state since we need to pass it to the sentinal later. 
            h_prev = h.clone()
            h, c = self.LSTM(current_input, (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_size)
            # Run the sentinal model
            st = self.sentinal(current_input, h_prev[:batch_size_t], c)

            # Run the adaptive attention model
            alpha_t, beta_t, c_hat = self.adaptive_attention(spatial_image[:batch_size_t],h,st)
            # Compute the probability over the vocabulary
            pt = self.fc(self.dropout(c_hat + h))                  # (batch_size, vocab_size)
            predictions[:batch_size_t, timestep, :] = pt
            alphas[:batch_size_t, timestep, :] = alpha_t
            betas[:batch_size_t, timestep, :] = beta_t
        return predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind  
