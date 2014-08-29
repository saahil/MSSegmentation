import numpy, Image
from scipy.stats import norm

def normfit(im, im_base):
    #convert image to array
    im = numpy.array(im)
    im_base = numpy.array(im_base)
    
    (mu,sigma) = norm.fit(im_base.flatten())
    print mu, sigma

    im = im - mu
    im = 255 * im / sigma

    im = Image.fromarray(im).convert('L')

    return im

