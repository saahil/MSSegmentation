import sys, os
from PIL import Image
import numpy
from tempfile import TemporaryFile
import random

IMAGE_SIZE = (1024,1224)
POPULATION_M = 3000
POPULATION_U = 9000
CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

def extract_point(ims, x, y):
    X = []
    
    for i in CHANNELS:
        #channel = im_name+'-'+str(i)+'.png'
        #im = Image.open(channel)
        #im = numpy.array(im)
        im = ims[i]
        X.append(im[x][y])

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
    
    # Check for rgb images
    if not os.path.isfile(im_name+'-0.png'):
        if not os.path.isfile(im_name+'-rgb.png'):
            print 'either ' + im_name + '-0.png or ' + im_name + '-rgb.png must exist'
            sys.exit('exiting...')
        else:
            im = Image.open(im_name+'-rgb.png')
            im = numpy.array(im)
            im = numpy.swapaxes(im, 1, 2)
            im = numpy.swapaxes(im, 0, 1)
            for i in range(3):
                ch = im[i].copy()
                ch = Image.fromarray(ch)
                ch.save(im_name+'-'+str(i)+'.png')
    # Optimizing by only opening image files once
    ims = []
    for i in CHANNELS:
        channel = im_name+'-'+str(i)+'.png'
        im = Image.open(channel)
        
        # contrast normalization
        #im = histeq(im)
        
        im = numpy.array(im)
        ims.append(im)

    # extract pixels from sampled points
    print 'extracting pixels from - '+im_name+' ...'
    print len(sampling)
    for i,s in enumerate(sampling):
        extracted = extract_point(ims, s[0], s[1])
        if extracted==[]:
            print 'sampling error at ' + str(s) + '\nExiting...'
            quit(-1)
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

