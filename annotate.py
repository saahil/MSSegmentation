from PIL import Image
import utils.patches as patches
import numpy, sys

test_path = sys.argv[1]
out_path = sys.argv[2]
chan_4 = Image.open(test_path+'-4.png')
chan_4 = chan_4.resize((1224,1024))
img_ar = numpy.array(chan_4)
im_cp = numpy.array(img_ar, dtype='uint8')
i = patches.PATCH_SIZE[0]/2+1
j = patches.PATCH_SIZE[1]/2+1

#y = numpy.empty((1,0), dtype='uint8')
y = []
for fi in range(0,226):
    cur_y = numpy.load(out_path+test_path.split('/')[-1]+'_'+str(fi)+'.npy')
    y = numpy.concatenate((y, cur_y))

loop_out = False
k=0
while i<(patches.IMAGE_SIZE[0]-patches.PATCH_SIZE[0]/2):
    while j<(patches.IMAGE_SIZE[1]-patches.PATCH_SIZE[1]/2):
        im_cp[i][j] = img_ar[i][j]
        if y[k]==0:
            im_cp[i][j] = 200
        k += 1
        if (k==len(y)):
            loop_out = True
            break
        j += 1
    if loop_out==True:
        break
    i += 1
    j = patches.PATCH_SIZE[1]/2+1

#for fi in range(0,226):
#    loop_out = False
#    cur_y = numpy.load(out_path+test_path.split('/')[-1]+'_'+str(fi)+'.npy')
#    p = 0
#    while i<(patches.IMAGE_SIZE[0]-patches.PATCH_SIZE[0]/2):
#        while j<(patches.IMAGE_SIZE[1]-patches.PATCH_SIZE[1]/2):
#            if cur_y[p]==0:
#                im_cp[i][j] = 200
#            p += 1
#            if p==4999:
#                loop_out = True
#                break
#            j += 1
#        if loop_out==True:
#            break
#        i += 1
#        j = patches.PATCH_SIZE[1]/2 + 1

im_cp = Image.fromarray(im_cp.astype(numpy.uint8))
im_cp.save(out_path+test_path.split('/')[-1]+'annotated.png')
