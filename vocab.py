import os
import json
from collections import Counter

#Load Karpathy's Split. Note that this split is different from the train,val,test split of the original dataset split
#Here, we will use the Flickr8k dataset. You can use the other datasets if you have enough memory for the h5py to be stored. 
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
