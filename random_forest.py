import os, sys
import numpy
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import utils.pixels as pixels
import utils.patches as patches
import utils.features as features
import utils.long_features as long_features
import utils.hist as hist
from PIL import Image

N_CHANNELS = 10

# For pixel inputs
#N_SAMPLING = 12000
#N_AROUND = 1
#IMAGE_SIZE = pixels.IMAGE_SIZE

# For patch inputs or patch histogram features
N_SAMPLING = 8000
N_AROUND = 256
IMAGE_SIZE = hist.IMAGE_SIZE
PATCH_SIZE = hist.PATCH_SIZE
CHANNELS = hist.CHANNELS

N = N_SAMPLING*17
N_TRAIN = (N/4)*3
N_VALID = N/4

TEST_IMAGES = ['../data/PatientMS/20140120T143753/20140120T143753', '../data/PatientMS/20140120T150515/20140120T150515']

def extract_patches_from_all_points(im_name):
    X = []
    print "extracting all patches"
    if not os.path.exists(im_name+'_0.npy'):
        #i = 384 + 28
        #j = 464 + 28
        file_index = 0;
        n_loaded = 0;
        split_file = open('split.txt', 'w+')
        
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
            im = numpy.array(im)
            ims.append(im)
        
        i = hist.PATCH_SIZE[0]/2+1
        while i<(hist.IMAGE_SIZE[0]-hist.PATCH_SIZE[0]/2):
            j = hist.PATCH_SIZE[1]/2+1
            while j<(hist.IMAGE_SIZE[1]-hist.PATCH_SIZE[1]/2):
                temp_X = hist.extract_patch(ims, i, j)
                if temp_X==[]:
                    print "error loading patch at: " + str(i) + ", " + str(j)
                X.append(temp_X)
                n_loaded += 1
                if (n_loaded%8000 == 0):
                    numpy.save(im_name+'_'+str(file_index), X)
                    print "saved file: " + str(file_index)
                    X = []
                    split_file.write('\n'+str(file_index)+': '+str(i)+', '+str(j))
                    file_index += 1
                j += 1
            i += 1
        
        if (n_loaded%8000 != 0):
            numpy.save(im_name+'_'+str(file_index), X)
            split_file.write('\n'+str(file_index)+': '+str(i)+', '+str(j))
        split_file.close()

if __name__=="__main__":
    data_path = sys.argv[1]
    X = numpy.zeros((N,N_CHANNELS*N_AROUND), dtype='float32')
    y = numpy.zeros((N), dtype='uint8')
    
    i = 0
    for fl in glob.glob(data_path+'*_X.npy'):
        temp_X = numpy.load(fl)
        X[i*N_SAMPLING:(i+1)*N_SAMPLING, :] = temp_X
        i += 1

    i = 0
    for fl in glob.glob(data_path+'*_Y.npy'):
        temp_y = numpy.load(fl)
        y[i*N_SAMPLING:(i+1)*N_SAMPLING] = temp_y
        i += 1

    # shuffle
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(X)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(y)

    #split into training and validation sets
    X_train = X[0:N_TRAIN]
    y_train = y[0:N_TRAIN]

    X_valid = X[N_TRAIN:N_TRAIN+N_VALID]
    y_valid = y[N_TRAIN:N_TRAIN+N_VALID]
    
    print X_train.shape
    print y_train.shape
    print X_valid.shape
    print y_valid.shape

    print 'training randomized forests'
    clf = RandomForestClassifier(n_estimators=150, criterion='gini', max_features=100, max_depth=2000)
    clf = clf.fit(X_train, y_train)

    print 'validation errors'
    valid_scores = cross_val_score(clf, X_valid, y_valid)
    print valid_scores.mean()

    # segment test images
    for im_name in TEST_IMAGES:
        ims = []
        X = []
        y = []
        print 'Predicting labels for ' + im_name
        
        if N_CHANNELS>4:
            im_orig = Image.open(im_name+'-4.png')
        else:                                                                                                                                                                                                                            
            im_orig = numpy.zeros(IMAGE_SIZE, dtype='uint8')
        im_orig = numpy.array(im_orig)
        im_new = numpy.zeros(im_orig.shape, dtype='uint8')
        
        # For pixel inputs
        #for x in range(IMAGE_SIZE[0]):
        #    for y in range(IMAGE_SIZE[1]):
        #        X.append(pixels.extract_point(ims, x, y))
        
        #y = y.reshape((1024,1224))
        #for i in range(IMAGE_SIZE[0]):
        #    for j in range(IMAGE_SIZE[1]):
        #        im_new[i][j] = im_orig[i][j]
        #        if y[i][j]==1:
        #            im_new[i][j] = 200
        
        # For patch inputs or patch histogram
        if not os.path.exists(im_name+'_0.npy'):
            extract_patches_from_all_points(im_name) #save patches in batches
        
        fi = 0
        while(True):
            fl = im_name + "_" + str(fi) + ".npy"
            if not os.path.isfile(fl):
                break
            X = numpy.load(fl)
            X = numpy.float32(X)
            Y = clf.predict(X)
            Y = numpy.array(Y)
            y = numpy.concatenate((y, Y))
            fi += 1
        
        numpy.save(im_name+'_y.npy', y)
        print y.shape
        
        loop_out = False
        k=0
        i = PATCH_SIZE[0]/2+1
        
        while i<(IMAGE_SIZE[0]-PATCH_SIZE[0]/2):
            j = PATCH_SIZE[1]/2+1
            while j<(IMAGE_SIZE[1]-PATCH_SIZE[1]/2):
                im_new[i][j] = im_orig[i][j]
                if y[k]==1:
                    im_new[i][j] = 200
                k += 1
                if (k==len(y)):
                    loop_out = True
                    break
                j += 1
            if loop_out==True:
                break
            i += 1
        
        numpy.save(im_name+'_im_rf.npy', im_new)
        im_out = Image.fromarray(im_new)
        im_out.save(im_name+'_annotated_rf.png')
