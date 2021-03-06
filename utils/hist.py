import sys, os
from PIL import Image
import numpy
from tempfile import TemporaryFile
import random

PATCH_SIZE = (45, 45)
IMAGE_SIZE = (1024,1224)
POPULATION_M = 3000
POPULATION_U = 5000
CHANNELS = (0, 1, 2)

def extract_patch(ims, x, y):
    X = []
    
    #If the patch extraction point is too close to the image border
    if x<=PATCH_SIZE[0]/2 or y<=PATCH_SIZE[1]/2 or x>=IMAGE_SIZE[0]-PATCH_SIZE[0]/2 or y>=IMAGE_SIZE[1]-PATCH_SIZE[1]/2:
        return X

    for i in CHANNELS:
        #channel = im_name+'-'+str(i)+'.png'
        #im = Image.open(channel)
        #im = numpy.array(im)
        im = ims[i]
        temp_patch = []
        for cur_x in xrange(x-PATCH_SIZE[0]/2, x+PATCH_SIZE[0]/2+1):
            for cur_y in xrange(y-PATCH_SIZE[1]/2, y+PATCH_SIZE[1]/2+1):
                temp_patch.append(im[cur_x][cur_y])
        temp_patch = numpy.array(temp_patch, dtype='uint8')
        #temp_patch = Image.fromarray(temp_patch.reshape(PATCH_SIZE))
        #X = numpy.concatenate((X, temp_patch.histogram()))
        X = numpy.concatenate((X, numpy.histogram(temp_patch, bins=256, range=(0., 256.))[0]))

    return X

if __name__=='__main__':
    if len(sys.argv)!=3:
        print 'usage: patches.py image_name mask_name'
        quit(-1)

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

    marked = []
    around = []

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
        channel = im_name+'-'+str(i)+'.png' # change to '.png' maybe
        im = Image.open(channel)
        im = numpy.array(im)
        ims.append(im)

    # extract patches from sampled points
    print 'extracting patches from - '+im_name+' ...'
    print len(sampling)
    for i,s in enumerate(sampling):
        extracted = extract_patch(ims, s[0], s[1])
        if extracted==[]:
            print 'sampling error at ' + str(s) + '\nExiting...'
            quit(-1)
        X.append(extracted)
        if mask[s[0]][s[1]] == 0:
            Y.append(1)
        else:
            Y.append(0)

    print 'saving patches ...'
    X = numpy.array(X)
    Y = numpy.array(Y)
    print X.shape
    numpy.save(np_file, X)
    numpy.save(np_file_y, Y)

