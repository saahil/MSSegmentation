import numpy, Image
import sys
import random

PATCH_SIZE = (55,55)
IMAGE_SIZE = (1024,1224)
POPULATION_M = 3000
POPULATION_U = 5000
CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

if len(sys.argv) != 2:
    print 'usage: sampling.py mask_file_name'
    quit(-1)

mask_name = sys.argv[1]
mask = Image.open(mask_name)
mask = numpy.array(mask)

marked = []
unmarked = []

x = int(PATCH_SIZE[0]/2) + 1
while x<int(IMAGE_SIZE[0]-PATCH_SIZE[0]/2-1):
    y = int(PATCH_SIZE[1]/2) + 1
    while y<int(IMAGE_SIZE[1]-PATCH_SIZE[1]/2-1):
        if mask[x][y] == 0:
            marked.append((x,y))
        else:
            unmarked.append((x,y))
        y += 1
    x += 1

# sample marked points
print "sampling lesion points..."
marked_sample = random.sample(marked, POPULATION_M)

#sample unmarked points
print "sampling background points..."
unmarked_sample = random.sample(unmarked, POPULATION_U)

# concatenate both
sample = numpy.concatenate((marked_sample,unmarked_sample))

# save sample
print "saving..."
sample = numpy.array(sample, dtype='uint16') # smallest datatype that can hold upto 1224
print sample
print sample.dtype
#sample_file = open(mask_name[:-4]+'_sample.txt', 'w+')
#numpy.savetxt(mask_name[:-4]+'_sample.txt', sample)
numpy.save(mask_name[:-4]+'_' + str(POPULATION_U+POPULATION_M) + '_sample.npy', sample)
#sample_file.write(sample)
