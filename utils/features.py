import sys
from PIL import Image
import numpy
from tempfile import TemporaryFile
import random
from skimage.feature import local_binary_pattern

IMAGE_SIZE = (1024,1224)
POPULATION_M = 3000
POPULATION_U = 9000
CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
P = [4, 8]
R = [1, 2, 3, 4, 5]

def extract_features(ims):
    X = []
    lbp_ar = []

    for im in ims:
        for p in P:
            for r in R:
                lbp_ar.append(local_binary_pattern(im, p*r, r, method='uniform'))
    X = lbp_ar[0]
    for lbp in lbp_ar[1:]:
        X = numpy.dstack((X, lbp))
    print X.shape
    return X

if __name__=='__main__':
    im_name = sys.argv[1]
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
    features = extract_features(ims)

    for i,s in enumerate(sampling):
        extracted = features[s[0]][s[1]]
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

