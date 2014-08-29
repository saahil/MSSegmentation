import os
import numpy
import glob

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter

SHAPE = [55, 55, 10]
NSAMPLING = 8000
NIMAGES = 20
N = NIMAGES*NSAMPLING
NTRAIN = NSAMPLING*12
NTEST = NSAMPLING*4

class MSRDataset(DenseDesignMatrix):
    def __init__(self, path_to_data='/home/ognawala/data/PatientMS-R/', which_set='train', start=None, stop=None, axes=('b', 0, 1, 'c'), preprocessor=None):
        self.axes = axes
        self.path_to_data = path_to_data

        if which_set == 'train':
            X, y = self._load_data(path_to_data, True)
        if which_set == 'test':
            X, y = self._load_test_data(path_to_data, True)
        
        # define numbers

        #if isinstance(y,list):
        #    y = numpy.asarray(y)
        
        if start is not None:
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop]
            assert X.shape[0] == y.shape[0]

        view_converter = DefaultViewConverter(shape=SHAPE, axes=axes)

        super(MSRDataset,self).__init__(X=X, y=y, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))

        if preprocessor:
            preprocessor.apply(self)

    def _load_data(self, path, expect_labels):
        dtype = 'uint8'
        ntrain = NTRAIN
        ntest = NTEST

        self.img_shape = SHAPE[::-1] #just reversing the dimensions
        self.img_size = numpy.prod(self.img_shape)
        self.n_classes = 2
        
        x = numpy.zeros((ntrain+ntest, self.img_size), dtype=dtype)
        y = numpy.zeros((ntrain+ntest,2), dtype=dtype)

        for i,fl in enumerate(glob.glob(path+'*X.npy')):
            temp_x = numpy.load(fl)
            x[i*NSAMPLING:(i+1)*NSAMPLING, :] = temp_x
        
        i = 0
        for fl in glob.glob(path+'*Y.npy'):
            temp_y = numpy.load(fl)
            for cur_y in temp_y:
                classes = [0, 0]
                classes[cur_y] = 1
                y[i] = classes
                i += 1

        # WRONG: y needs to be a column vector, instead of array
        # y = y.reshape((ntrain+ntest,1))

        Xs = {
                'train' : x[0:ntrain],
                'test'  : x[ntrain:ntrain+ntest]
            }
        Ys = {
                'train' : y[0:ntrain],
                'test'  : y[ntrain:ntrain+ntest]
            }

        X = numpy.cast['float32'](Xs['train'])
        if expect_labels:
            y = Ys['train']
        # shuffle both X and y
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(X)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(y)

        #only using for debugging
        #x_file = open('/home/ognawala/pylearn/trainingData/X.npy', 'w+')
        #numpy.save(x_file, X)
        #y_file = open('/home/ognawala/pylearn/trainingData/y.npy', 'w+')
        #numpy.save(y_file, y)
        return X, y

    def _load_test_data(self, path, expect_labels):
        dtype = 'uint8'
        ntrain = NTRAIN
        ntest = NTEST

        self.img_shape = SHAPE[::-1]
        self.img_size = numpy.prod(self.img_shape)
        self.n_classes = 2
        
        x = numpy.zeros((ntrain+ntest, self.img_size), dtype=dtype)
        y = numpy.zeros((ntrain+ntest,2), dtype=dtype)

        for i,fl in enumerate(glob.glob(path+'*X.npy')):
            temp_x = numpy.load(fl)
            x[i*NSAMPLING:(i+1)*NSAMPLING, :] = temp_x
        
        i = 0
        for fl in glob.glob(path+'*Y.npy'):
            temp_y = numpy.load(fl)
            for cur_y in temp_y:
                classes = [0, 0]
                classes[cur_y] = 1
                y[i] = classes
                i += 1

        # WRONG: y needs to be a column vector, instead of array
        # y = y.reshape((ntrain+ntest,1))

        Xs = {
                'train' : x[0:ntrain],
                'test'  : x[ntrain:ntrain+ntest]
            }
        Ys = {
                'train' : y[0:ntrain],
                'test'  : y[ntrain:ntrain+ntest]
            }

        X = numpy.cast['float32'](Xs['test'])
        if expect_labels:
            y = Ys['test']
        # shuffle both X and y
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(X)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(y)
        return X, y
    
    def get_test_set(self):
        return MSRDataset(path_to_data=self.path_to_data, which_set='test', axes=self.axes)
        
