import sys

import utils.patches as patches 
import numpy
from scipy.misc import imsave
from PIL import Image
import glob
import os

def usage():
    print """usage: test.py model.pkl pathToTestImage outputPath
Where model.pkl contains a trained pylearn2.models.mlp.MLP object.
The script will make images in path with classified pixels"""

def extract_patches_from_all_points(im_name):
    X = []
    print "extracting all patches"
    if not os.path.exists(im_name+'_0.npy'):
        #i = 384 + 28
        #j = 464 + 28
        file_index = 0;
        n_loaded = 0;
        split_file = open('split.txt', 'w+')
        
        # Optimizing by only opening image files once
        ims = []
        for i in patches.CHANNELS:
            channel = im_name+'-'+str(i)+'.png'
            im = Image.open(channel)
            im = numpy.array(im)
            ims.append(im)
        
        i = patches.PATCH_SIZE[0]/2+1
        while i<(patches.IMAGE_SIZE[0]-patches.PATCH_SIZE[0]/2):
            j = patches.PATCH_SIZE[1]/2+1
            while j<(patches.IMAGE_SIZE[1]-patches.PATCH_SIZE[1]/2):
                temp_X = patches.extract_patch(ims, i, j)
                if temp_X==[]:
                    print "error loading patch at: " + str(i) + ", " + str(j)
                X.append(temp_X)
                n_loaded += 1
                if (n_loaded%5000 == 0):
                    numpy.save(im_name+'_'+str(file_index), X)
                    print "saved file: " + str(file_index)
                    X = []
                    split_file.write('\n'+str(file_index)+': '+str(i)+', '+str(j))
                    file_index += 1
                j += 1
            i += 1
        
        if (n_loaded%5000 != 0):
            numpy.save(im_name+'_'+str(file_index), X)
            split_file.write('\n'+str(file_index)+': '+str(i)+', '+str(j))
        split_file.close()

if len(sys.argv) != 4:
    usage()
    print "(You used the wrong # of arguments)"
    quit(-1)

_, model_path, test_path, out_path = sys.argv

from pylearn2.utils import serial
try:
    model = serial.load(model_path)
except Exception, e:
    usage()
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
    print e

from pylearn2.config import yaml_parse

dataset = yaml_parse.load(model.dataset_yaml_src)

#extract patches from all points in the image
if not os.path.exists(test_path+'_0.npy'):
    extract_patches_from_all_points(test_path) #save patches in batches

# y = []

#read batches of patches and output labels to another file
for fi in range(0,226):
    fl = test_path + "_" + str(fi) + ".npy"
    print "Processing file: " + fl
    x = numpy.load(fl)
    # use smallish batches to avoid running out of memory
    batch_size = 100
    model.set_batch_size(batch_size)
    # dataset must be multiple of batch size of some batches will have
    # different sizes. theano convolution requires a hard-coded batch size
    m = x.shape[0]
    #extra = batch_size - m % batch_size
    #assert (m + extra) % batch_size == 0
    
    #if extra > 0:
    #    x = numpy.concatenate((x, numpy.zeros((extra, x.shape[1]),
    #                                                    dtype=x.dtype)), axis=0)
    print x.shape
    assert x.shape[0] % batch_size == 0
    
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    
    from theano import tensor as T
    
    y = T.argmax(Y, axis=1)
    
    from theano import function
    
    f = function([X], y)
    
    print "forward propagating.."
    y = []
    for i in xrange(x.shape[0] / batch_size):
        x_arg = x[i*batch_size:(i+1)*batch_size,:]
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        y.append(f(x_arg.astype(X.dtype)))
    
    y = numpy.concatenate(y)
    assert y.ndim == 1
    assert y.shape[0] == x.shape[0]
    y = numpy.abs(y[:m] - 1)
    
    #save y with the same name in output_folder
    numpy.save(out_path+test_path.split('/')[-1]+'_'+str(fi)+'.npy', y)
    
    print y.shape
    
    #m = 0
    #nm = 0
    
    #for l in y:
    #if l==1:
    #    m += 1
    #else:
    #    nm += 1
    # 
    #print m, nm

