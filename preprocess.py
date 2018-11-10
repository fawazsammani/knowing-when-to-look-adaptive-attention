import os
import json
from collections import Counter
import h5py
from random import seed, choice, sample
from tqdm import tqdm
from scipy.misc import imread, imresize


#Load Karpathy's Split. 
with open('dataset_flickr30k.json', 'r') as j:       
    data = json.load(j)

#Define the lists to store the images paths and the captions of the image
train_image_paths = []
train_image_captions = []
val_image_paths = []
val_image_captions = []
test_image_paths = []
test_image_captions = []
word_freq = Counter()

captions_per_image=5
min_word_freq=5
max_len=50

#Let's visualize how the data is arranged
i=0
for img in data['images']:
    if i==1: break
    print(img['filename'])
    for c in img['sentences']:
        print(c['tokens'])
        i=1
        

#Create the lists. The image_paths lists contains the path of every image, and the image_captions contains the corresponding
#captions (can be more than one) for each image. Therefore, the lengths of the two should be identical. 
for img in data['images']:                      #Start by looping through every image in Karpathy's split
    captions = []                               #Define a list to append the captions of that specific image
    for c in img['sentences']:                  #Loop through every sentence of the image
        # Update word frequency
        word_freq.update(c['tokens'])
        
        if len(c['tokens']) <= max_len:         #Make sure the sentence isn't too long, and ignore it if so
            captions.append(c['tokens'])        #Append the list of words of the sentence to the list. len(captions) = 5
    #Get the path of the image
    path = os.path.join('images', img['filename'])     #path if using the Flickr dataset

    if img['split'] in {'train', 'restval'}:     #Get what karpathy's split is
        train_image_paths.append(path)           #Append the image path to the image_paths list
        train_image_captions.append(captions)    #Append the captions list (of size 5) to the image_captions list

    elif img['split'] in {'val'}:
        val_image_paths.append(path)
        val_image_captions.append(captions)

    elif img['split'] in {'test'}:
        test_image_paths.append(path)
        test_image_captions.append(captions)

#Check that everything is OK
assert len(train_image_paths) == len(train_image_captions)
assert len(val_image_paths) == len(val_image_captions)
assert len(test_image_paths) == len(test_image_captions)


print("Words before filtering: ",len(word_freq.keys()))
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
print("Words after filtering: ",len(words))


#Create the word2index dictionary (maps a word to a unique index)
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0


#Save the wordmap file to drive
with open('WORDMAP.json', 'w') as j:
    json.dump(word_map, j)
    

for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                (val_image_paths, val_image_captions, 'VAL'),
                                (test_image_paths, test_image_captions, 'TEST')]:

    with h5py.File(os.path.join('caption data', split + '_IMAGES_' + '.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        h.attrs['captions_per_image'] = captions_per_image
        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 254, 254), dtype='uint8')
        print("\nReading {} images and captions, storing to file...\n".format(split))
        
        enc_captions = []
        caplens = []

        for i, path in enumerate(tqdm(impaths)):
            # Sample captions
            # If the image includes less than 5 captions, then complete it to 5 from any of the captions. Else, just shuffle 
            # the 5 captions to different order using random.sample
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)
            #Check that everything is OK
            assert len(captions) == captions_per_image
            
            # Read images
            img = imread(impaths[i]) 
            img = imresize(img, (254, 254))                     # Resize the image to the expected size
            img = img.transpose(2, 0, 1)                        # make the channels (index 2) first,as expected by PyTorch
            
            assert img.shape == (3, 254, 254)                   # raise an error if the image shape is not (3,256,256)
            # Save image to HDF5 file
            images[i] = img
            
            for c in captions:                                  # For every caption of the 5 captions
                #Encode captions
                #Use the dict.get(key, default = None)
                #key: This is the Key to be searched in the dictionary 
                #default: This is the Value to be returned in case key does not exist
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                # Find caption lengths without the padding 
                c_len = len(c) + 2
                enc_captions.append(enc_c)                      #Append the list of encoded caption to the list enc_captions
                caplens.append(c_len)                           #Append the caption length to the list c_len
                
        #Make sure everything is correct. Remember when we initialized images: images.shape[0] = len(impaths)        
        assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
        
        # Save encoded captions and their lengths to JSON files
        with open(os.path.join('caption data', split + '_CAPTIONS_' + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join('caption data', split + '_CAPLENS_' + '.json'), 'w') as j:
            json.dump(caplens, j)
