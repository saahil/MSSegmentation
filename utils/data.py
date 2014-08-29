#! /usr/bin/env python2.6
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer <bayer.justin@googlemail.com>'


import glob
import itertools
import operator
import os
import re

import scipy

#from arac.utilities import block_permutation
#from arac.cppbridge import SupervisedSimpleDataset


def volume_from_file(filename, frameshape=(512, 512)):
    with file(filename) as fp:
        arr = scipy.fromfile(fp, dtype=scipy.uint8)
    x, y = frameshape
    n, = arr.shape
    n /=  x * y
    arr.shape = n, x, y
    return arr


def mask_from_file(filename):
    """Return a pair (dimensions, dict) where dimensions is a 3-tuple containing
    the size of the three dimensions and dict is a mask dictionary with 
    mappings from mask-type-identifiers to sets that contain the segmented 
    coordinates."""
    header_pattern = re.compile(
        "\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)")
    splitter_pattern = re.compile("([\.a-zA-Z ]+),\s*([\w ]+),\s*(\d+)")
    segment_pattern = re.compile("\s*(\d+),\s*(\d+),\s*(\d+)")
    deadline_pattern = re.compile(",\s*([\w ]+),\s*(\d+)")
    
    def get_header(line):
        """Return a header dict if line is a header, otherwise None."""
        match = header_pattern.match(line)
        if match is None:
            return None
        keys = 'width', 'height', 'depth'
        return dict(zip(keys, match.groups()))
        
    def get_splitter(line):
        """Return a dict with keys identifier and amount if the line is a 
        splitter, otherwise None."""
        match = splitter_pattern.match(line)
        if match is None:
            return None
        identifier, dontknow, amount = match.groups()
        return dict(identifier=identifier, amount=amount)
    
    def get_segment(line):
        match = segment_pattern.match(line)
        if match is None:
            return None
        return match.groups()
        
    def deadline(line):
        """Tell wether the line is a known type of deadline, which can be 
        ignored."""
        return deadline_pattern.match(line) is not None
        
    result = {}
    
    with file(filename) as fp:
        lines = iter(fp.xreadlines())
        first_line = lines.next()
        header = get_header(first_line)
        lines.next()    # Ignore second line (don't know what it means)
        if header is None:
            raise IOError("Bad segmentation file.")
        shape = header['width'], header['height'], header['depth']
        shape = tuple(int(i) for i in shape)
        for i, line in enumerate(lines):
            splitter = get_splitter(line)
            if splitter is not None:
                current = set()
                result[splitter['identifier']] = current
            else:
                segment = get_segment(line)
                if segment is not None:
                    point = segment
                    x, y, z = tuple(int(i) for i in point)
                    current.add((y, x, z))
                elif not deadline(line):
                    raise IOError("Bad segmentation line #%i." % (i + 1))
                
    return shape, result
    

def array_from_mask(mask, (height, width, depth), masks=None):
    if masks is None:
        masks = 'Med. Tibia'
    arr = scipy.zeros((depth, width, height, len(masks) + 1), dtype=scipy.uint8)
    arr[:,:,:,-1] = scipy.ones((depth, width, height))
    for i, n in enumerate(masks):
        for x, y, z in mask[n]:
            arr[z - 1, x - 1, y - 1, i] = 1
            arr[z - 1, x - 1, y - 1, -1] = 0
    return arr
    
    
def maskarray_from_file(filename, masks=None):
    shape, mask = mask_from_file(filename)
    volume = array_from_mask(mask, shape, masks)
    return volume


class Repository(object):

  def __init__(self, path, blockshape=None, frame=None, slice=None):
    self.path = path
    self.blockshape = blockshape
    self.frame = frame
    self.slice = slice
    self.files = self._files()
    self.file_by_ident = dict(zip(self.idents(), self.files))

  def _files(self):
    return sorted(glob.glob(os.path.join(self.path, self.fileglob)))

  def idents(self):
    return [self.ident(f) for f in self.files]

  def ident(self, filename):
    return os.path.split(filename)[1][:6]

  def __iter__(self):
    """Return an iterator over the arrays in the volume."""
    for fn in self.files:
      yield self._by_file(fn)

  def __getitem__(self, key):
    return self._by_file(self.file_by_ident[key])

  def sliced(self, arr):
    if self.slice is not None:
      minx, maxx, miny, maxy = self.slice
    else:
      maxx = arr.shape[1] 
      maxy = arr.shape[2]
      minx, miny = 0, 0
    if self.frame is None:
      return arr[:, minx:maxx, miny, maxy]
    else:
      return arr[self.frame, minx:maxx, miny:maxy]


class VolumeRepository(Repository):

  extension = '.arr'
  fileglob = '*' + extension

  def _by_file(self, fn):
    vol = volume_from_file(fn)
    return self.sliced(vol).astype('float64') / 255 + 128

  def _files(self):
    cands = super(VolumeRepository, self)._files()
    # Filter out files that are not of the correct size.
    return [f for f in cands if os.path.getsize(f) == 512 * 512 * 48]


class MaskRepository(Repository):

  extension = ''
  fileglob = '*_Cor_qcfin'

  def __init__(self, path, blockshape=None, frame=None, slice=None, masks=None):
    self.masks = masks if masks is not None else ('Med. Tibia', )
    super(MaskRepository, self).__init__(path, blockshape, frame, slice)

  def _by_file(self, fn):
    arr = maskarray_from_file(fn, self.masks)
    return self.sliced(arr).astype('float64')


class DataAccessor(object):
  
  def __init__(self, volumepath, maskpath, 
               blockshape=None, frame=None, slice=None, masks=None):
    self.masks = masks if masks is not None else ('Med. Tibia', )
    self._blockshape = blockshape
    self._frame = frame
    self._slice = (0, 512, 0, 512) if slice is None else slice
    self.volrepos = VolumeRepository(volumepath, blockshape, frame=frame, 
                                     slice=slice)
    self.maskrepos = MaskRepository(maskpath, blockshape, frame=frame, 
                                    slice=slice, masks=self.masks)

  @property
  def seqlength(self):
    return reduce(operator.mul, self.shape, 1) / self.blocksize

  @property
  def blocksize(self):
    return reduce(operator.mul, self._blockshape, 1)

  @property
  def shape(self):
    minx, maxx, miny, maxy = self._slice
    return maxx - minx, maxy - miny

  def idents(self):
    maskidents = set(self.maskrepos.idents())
    volidents = set(self.volrepos.idents())
    return maskidents & volidents

  def __iter__(self):
    for ident in self.idents():
      yield self.volrepos[ident], self.maskrepos[ident]

  def blockpermuted(self):
    for vol, mask in self:
      # This is serious stuff. Don't change it. If you want to, first make
      # some intelligent comments how this works.
      vol.shape = self.seqlength, self.blocksize

      numclasses = len(self.masks) + 1
      mask = blockpermute(mask, self.shape, self._blockshape, 
                          itemsize=numclasses)
      mask.shape = self.seqlength, self.blocksize, numclasses 
      mask = mask.max(axis=1)

      # If blocks are part of any mask, their no-mask part is explicitly set to
      # zero.
      for row in mask:
        if (row[:-1] == 1.).any():
          row[-1] = 0.
        row /= row.sum()

      mask.shape = (numclasses, 
                    self.shape[0] / self._blockshape[0], 
                    self.shape[1] / self._blockshape[1])

      yield vol, mask
      

def take(iterable, n):
  return itertools.islice(iterable, n)


def blockpermute(arr, shape, blockshape, itemsize=1, invert=False):
  """
  First test an array without classes.
  >>> arr = scipy.array(range(16))
  >>> blockpermute(arr, (4, 4), (2, 2))
  array([ 0,  1,  4,  5,  2,  3,  6,  7,  8,  9, 12, 13, 10, 11, 14, 15])
  >>> a = [0, 1]
  >>> b = [1, 0]
  >>> arr = scipy.array(2 * a + 2 * b + 2 * a + 10 * b)
  >>> blockpermute(arr, (4, 4), (2, 2), itemsize=2)
  array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         0, 1, 0, 1, 0, 1, 0, 1, 0])

  """
  blocksize = reduce(operator.mul, blockshape, 1)
  sequencesize = reduce(operator.mul, shape, 1) / blocksize
  permutation = block_permutation(shape, blockshape)
  arr = arr.reshape(sequencesize * blocksize, itemsize)
  if invert:
      identity = range(reduce(operator.mul, shape, 1))
      permutation = [permutation[i] for i in identity]
  permuted = scipy.array([arr[i] for i in permutation]).flatten()
  return permuted


_dontkill = []


def normalize(iterable):
  """Normalize the samples of an accessor iterable.
  
  >>> x = [(scipy.array((1., 2., 3., 2.)), 1),  (scipy.array((4., -3., 2., 1.)), 1)]
  >>> list(normalize(x))
  [(array([ 0.14285714,  0.42857143,  0.71428571,  0.42857143]), 1), (array([ 1.        , -1.        ,  0.42857143,  0.14285714]), 1)]
  """
  lst = list(iterable)
  maxi = max(i.max() for i, _ in lst)
  mini = min(i.min() for i, _ in lst)
  # In order to not keep the whole list in memory but consume it, reverse
  # the list and take the last (formerly first) element out of it.
  lst.reverse()
  maxmindiffby2 = (maxi - mini) / 2
  while lst:
    sample, target = lst.pop()
    sample = (sample - mini - maxmindiffby2) / maxmindiffby2
    yield sample, target


def rectangle(iterable):
  """Turn the masks of an accessor iterable into the coordinates of an rectangle
  surrounding all the segements points.

  The coordinates are normalized to the interval [-1, 1]."""
  # TODO: make this work for multiclass data.
  for sample, target in iterable:
    orig_shape = target.shape[1], target.shape[2]
    target.shape = scipy.size(target) / 2, 2
    classes = target[:, 0].copy()
    classes.shape = orig_shape

    indices = scipy.where(classes.sum(axis=0) >= 1)
    min0, max0 = indices[0][0], indices[0][-1]
    indices = scipy.where(classes.sum(axis=1) >= 1)
    min1, max1 = indices[0][0], indices[0][-1]
    print min0, max0, min1, max1

    # Normalize.
    normalize = lambda x, rng: 2. * x / rng - 1
    size0, size1 = classes.shape[0], classes.shape[1]
    min0 = normalize(min0, size0)
    max0 = normalize(max0, size0)
    min1 = normalize(min1, size1)
    max1 = normalize(max1, size1)

    target = scipy.array((min0, max0, min1, max1))
    print target
    yield sample, target


def aracds(iterable, conf):
  # Construct dataset and return it.
  ds = SupervisedSimpleDataset(conf.insize, conf.outsize)

  # Hack references for the array so they are not deleted.
  for s, t in iterable:
    ds.append(s.ravel(), t.ravel())
    _dontkill.append((s, t))
  return ds


def attachimportance(ds, conf):
  for i in xrange(ds.size()):
    target = ds.target(i)
    target = target.reshape(conf.seqsize, conf.numclasses)
    count = target.sum(axis=0) + 1 # Avoid 0's / LaPlace's rule.
    prior = count / target.sum()
    importance = 1 / prior
    # Normalize.
    importance /= importance.sum() / conf.numclasses
    # Build up array.
    importance = scipy.tile(importance, conf.seqsize)
    _dontkill.append(importance)
    ds.set_importance(i, importance)


if __name__ == '__main__':
  import doctest
  doctest.testmod()
