import os
import numpy as np
import json
import sys
from PIL import Image
from matplotlib import pyplot as plt
import time
import random
def detect_red_light(I, prototype, channel_sel = 'red', thres = 0.90, method = 'inner_product'):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''

    # Generate selected channels
    channel_choice = {'red':0,'green':1,'blue':2}
    if channel_sel == 'all':
        channels = [0,1,2]
    else:
        channels = [channel_choice.get(channel_sel)]




    # Get all possible boxes
    height = prototype.shape[0]
    width = prototype.shape[1]
    all_boxes, box_ids = get_boxes(height,width, I, channels)



    # Compute 'distance' between prototype and each box with all the channels selected
    matched = np.zeros(all_boxes.shape[1])
    for channel in range(len(channels)):
        for box_idx in range(all_boxes.shape[1]):
            I_box = np.copy(all_boxes[channel, box_idx])
            matched[box_idx] += match(prototype[:,:,channel], I_box, method)

    if method == 'inner_product':
        matched = matched/len(channels)
        bounding_boxes = []
        for box_idx in range(all_boxes.shape[1]):
            if matched[box_idx] >= thres:
                bounding_boxes.append(box_ids[box_idx].astype(int).tolist())
                #I = visualize_box(box=box_ids[box_idx], image=I)
        token = random.randint(0,1000000)
        address = '../data/images/result/yea' + str(token) + '.jpg'
        visualize(I,address)
        #bounding_boxes = np.asarray(bounding_boxes)
        #print(len(bounding_boxes))

    if method == 'square_diff':
        matched = np.sqrt(matched)
        bounding_boxes = []
        for box_idx in range(all_boxes.shape[1]):
            if matched[box_idx] <= thres:
                bounding_boxes.append(box_ids[box_idx].astyp(int).tolist())
                #I = visualize_box(box=box_ids[box_idx], image=I)
        #token = random.randint(0,1000000)
        #address = '../data/images/template1_ip/yea_all' + str(token) + '.jpg'
        #visualize(I,address)
        #bounding_boxes = np.asarray(bounding_boxes)
        #print(len(bounding_boxes))

    #print(all_boxes.shape)
    #print(matched)
    #plt.plot(np.arange(1,len(matched)+1).astype(int),matched)
    #plt.show()
    #boxes = box_ids[np.argmax(matched)]
    #print(np.max(matched))
    #I_new = visualize_box(box = boxes, image = I)
    #visualize(I_new)




    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes


def normalization(I):

    '''
    Normalize I. Demean and set the std to 1.
    I: 2D np array (height x width)

    return 2D np array (height x width)
    '''


    I_new = np.copy(I).astype(float)
    sum = np.linalg.norm(I_new)
    if sum > 0:
        I_new = I_new/sum
    return I_new



def get_boxes(height, width, image, channels, step = 3):
    '''
    Given the width and height of a box, generate all the possible boxes.
    The width and height (integer) are that of the prototype.
    The image (np array) is the whole three channels of an image
    The channels (list) is all the channels we need to obtain within these boxes.
    E.G. channels = [0] means that only red channel is obtained.

    Return all_boxes and box_ids
    all_boxes: a 4D numpy array with shape (3,x,width,height) where x refers to the total # of possible boxes
    box_ids: the four vertices of a certain box
    '''
    img_height = image.shape[0]
    img_width = image.shape[1]
    all_boxes = []
    box_ids = []
    for channel in channels:
        channel_boxes = []
        for h in range(0, img_height-height+1, step):
            for w in range(0, img_width - width + 1, step):
                channel_boxes.append(normalization(np.array(image[h:(h+height),w:(w+width),channel])))
                box_ids.append([h,w,h+height,w+width])
        all_boxes.append(np.copy(np.asarray(channel_boxes)))

    return np.asarray(all_boxes), np.asarray(box_ids)


def match(prototype, box_I, method = 'inner_product'):
    '''
        Compare the 'distance' between image boxes and prototype
        Method: inner_product or square_diff
        Prototype: 2D np array (height x width)
        box_I: 2D np array (height x width)

        Return a float
    '''
    height = prototype.shape[0]
    width = prototype.shape[1]
    prototype_reshaped = np.reshape(prototype,height*width)
    box_I_reshaped = np.reshape(box_I,height*width)
    if method == 'inner_product':
        return np.dot(prototype_reshaped,box_I_reshaped)
    if method == 'square_diff':
        return np.sum((prototype_reshaped-box_I_reshaped)**2)


def visualize(I, address = '../data/images/test/yea.jpg'):
    '''
    Visualize the picture from 3D np array I
    '''
    img = Image.fromarray(I)
    #img.show()
    img.save(address)
    pass

def visualize_box(box, image, thick = 5):

    '''
    Visualize the bounding boxes with thick green sides
    Code for Q5 and Q6

    box: the bounding box coordinates, list of np arrays
    image: the 3D np array representing the original image (480 x 640 x 3)
    thick: how thick (how many pixels) the frame is

    This function returns a 3D np array representing the new image with bounding boxes
    '''
    height = image.shape[0]
    width = image.shape[1]
    I_new = np.copy(image)
    t = max(box[0]-thick,0)
    b = min(box[2]+thick,height)
    l = max(box[1]-thick,0)
    r = min(box[3]+thick,width)
    for h in range(t,box[0]):
        I_new[h, l:r, 0] = 0
        I_new[h, l:r, 1] = 255
        I_new[h, l:r, 2] = 0
    for h in range(box[0],box[2]):
        I_new[h, l:box[1], 0] = 0
        I_new[h, l:box[1], 1] = 255
        I_new[h, l:box[1], 2] = 0
        I_new[h, box[3]:r, 0] = 0
        I_new[h, box[3]:r, 1] = 255
        I_new[h, box[3]:r, 2] = 0
    for h in range(box[2],b):
        I_new[h, l:r, 0] = 0
        I_new[h, l:r, 1] = 255
        I_new[h, l:r, 2] = 0
    return I_new




# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = '../data/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]



### Create and normalize prototype
# Get prototype from RL009.jpg 1st
prototype = np.asarray(Image.open(os.path.join(data_path,file_names[10])))[68:130,350:378]

# Get prototype from RL020.jpg 2nd
#prototype = np.asarray(Image.open(os.path.join(data_path,file_names[20])))[147:170,280:290]

#Normalize the protype
prototype_n = np.copy(prototype).astype(float)
for i in range(3):
    prototype_n[:, :, i] = normalization(prototype[:, :, i])




### Gather information on test pictures
'''
preds = {}
test_id = [i for i in range(13)] + [i  for i in range(50,60)] + [i for i in range(166,173)]
for i in test_id:
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[9]))

    # convert to numpy array:
    I = np.asarray(I)
    preds[file_names[i]] = detect_red_light(I, prototype=prototype_n, channel_sel='all', method='inner_product', thres=0.92)
'''



preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    ts = time.time()
    preds[file_names[i]] = detect_red_light(I, prototype=prototype_n, channel_sel='red', method='inner_product', thres=0.92)
    te = time.time()
    print('Finish Image NO.%i; time elapsed: %.2f s'%(i, te-ts))



# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
