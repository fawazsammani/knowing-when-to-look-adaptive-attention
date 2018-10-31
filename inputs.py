import os
import h5py
import json
from tqdm import tqdm
from scipy.misc import imread, imresize
from vocab import *

for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                (val_image_paths, val_image_captions, 'VAL'),
                                (test_image_paths, test_image_captions, 'TEST')]:

    with h5py.File(os.path.join('caption data', split + '_IMAGES_' + '.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        h.attrs['captions_per_image'] = captions_per_image
        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
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
            img = imresize(img, (256, 256))                     # Resize the image to the expected size
            img = img.transpose(2, 0, 1)                        # make the channels (index 2) first,as expected by PyTorch
            
            assert img.shape == (3, 256, 256)                   # raise an error if the image shape is not (3,256,256)
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