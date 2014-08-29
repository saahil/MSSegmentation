import sys, os
from PIL import Image
import numpy
from tempfile import TemporaryFile
import random
from skimage.feature import local_binary_pattern

IMAGE_SIZE = (1024,1224)
POPULATION_M = 3000
POPULATION_U = 9000
CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
N_FEATURES = 32

def read_features(bz_name):
    os.system('bunzip2 -k %s'%bz_name)
    csv = open(bz_name[:-4], 'r')
    features = {}

    for i, line in enumerate(csv):
        if i==0:
            continue
        parts = line.strip().split(', ')
        coords = (int(float(parts[2])), int(float(parts[3])), int(parts[1][-6]))
        vector = parts[4:36]
        vector[31] = vector[31].split('.')[0]
        vector = [int(float(i)) for i in vector]
        features[coords] = vector

    os.system('rm -f %s'%bz_name[:-4])
    return features

def extract_features(features, x, y):
    X = []
    for i in xrange(10):
        X = numpy.concatenate((X, features[(y, x, i)])) # numpy arrays have inverted dimensions compared to PIL images

    return X

if __name__=='__main__':
    im_name = sys.argv[1]
    bz_name = im_name+'.csv.bz2'
    mask_name = sys.argv[2]
    sample_name = mask_name[:-4]+'_'+str(POPULATION_M+POPULATION_U)+'_sample.npy'
    
    np_file = open(im_name+'_X.npy', 'w+')
    np_file_y = open(im_name+'_Y.npy', 'w+')
    mask = Image.open(mask_name)
    mask = numpy.array(mask)
    sampling = numpy.load(sample_name)

    X = []
    Y = []

    # Optimizing by only opening image files once
    ims = []
    for i in CHANNELS:
        channel = im_name+'-'+str(i)+'.png'
        im = Image.open(channel)
        
        # contrast normalization
        #im = histeq(im)
        ims.append(im)

    # extract pixels from sampled points
    print 'extracting features from - '+im_name+' ...'
    print len(sampling)
    features = read_features(bz_name)

    for i,s in enumerate(sampling):
        extracted = extract_features(features, s[0], s[1])
        X.append(extracted)
        if mask[s[0]][s[1]] == 0:
            Y.append(1)
        else:
            Y.append(0)

    X = numpy.array(X)
    Y = numpy.array(Y)
    print X.shape
    numpy.save(np_file, X)
    numpy.save(np_file_y, Y)

