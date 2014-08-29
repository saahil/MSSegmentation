from PIL import Image

file = open("imgs.txt")
src = "/home/ognawala/pylearn/test_images/"
dst = "/home/ognawala/pylearn/test_images/downsized/"

for line in file:
    for i in range(0,10):
        im = Image.open(src+line.strip()+"-"+str(i)+".png")
        im = im.resize((1224,1024), Image.ANTIALIAS)
        im.save(dst+line.strip()+"-"+str(i)+".png")

#src = "/home/ognawala/data/masks/"
#dst = "/home/ognawala/data/masks_down/"

#file = open("/home/ognawala/data/imgs2.txt")
#for line in file:
#    im = Image.open(src+line.strip())
#    im = im.resize((1224,1024), Image.ANTIALIAS)
#    im.save(dst+line.strip())

