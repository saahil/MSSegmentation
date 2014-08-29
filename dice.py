import numpy
from PIL import Image
import sys

#if len(sys.argv) != 3:
#    sys.exit('usage: dice.py path_to_segmented_image path_to_ground_truth_image')

pairs = [['/home/ognawala/data/PatientMS-R/20140120T143753/20140120T143753_annotated_rf.png', '/home/ognawala/data/Patient-Mask/20140120T143753-mask.png'], ['/home/ognawala/data/PatientMS-R/20140120T150515/20140120T150515_annotated_rf.png', '/home/ognawala/data/Patient-Mask/20140120T150515-mask.png']]

# intersection set
n_aib = 0

#individual markings
n_y = 0
n_truth = 0

for p in pairs:
    y = Image.open(p[0])
    y = numpy.array(y, dtype='uint8')
    print p[0]
    print y.shape
    
    truth_im = Image.open(p[1])
    truth_y = numpy.array(truth_im, dtype='uint8')
    print p[1]
    print truth_y.shape
    
    # flatten arrays
    truth_y = truth_y.flatten()
    y = y.flatten()
    print truth_y.shape
    print y.shape
    
    for i in range(len(y)):
        # both marked?
        if y[i]==200 and truth_y[i]==0:
            n_aib += 1
        # y marked
        if y[i]==200:
            n_y += 1
        # truth marked
        if truth_y[i]==0:
            n_truth += 1

dice = float(2*n_aib)/float(n_y+n_truth)
print dice

