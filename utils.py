from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import gzip
import numpy as np
from skimage.io import imsave

LOG_MINIMUM = 1e-8

def _maybe_download(target_dir):
  target_path = os.path.join(target_dir, "mnist.pkl.gz")
  if not os.path.exists(target_dir):
    os.system(" ".join([
        "wget -P",
        target_dir,
        "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    ]))

def load_mnist(path):
  _maybe_download(os.path.dirname(path))
  file_in = gzip.open(path, "rb")
  if six.PY2:
    import cPickle
    train, valid, test = cPickle.load(file_in)
  elif six.PY3:
    import pickle
    u = pickle._Unpickler(file_in)
    u.encoding = 'latin1'
    train, valid, test = u.load()
  images = np.vstack([train[0], valid[0], test[0]])
  return images

def load_params(load_dir):
  b = np.load(os.path.join(load_dir, "b.npy"))
  c = np.load(os.path.join(load_dir, "c.npy"))
  w = np.load(os.path.join(load_dir, "w.npy"))
  return b, c, w

def save_params(save_dir, b, c, w):
  np.save(os.path.join(save_dir, "b.npy"), b)
  np.save(os.path.join(save_dir, "c.npy"), c)
  np.save(os.path.join(save_dir, "w.npy"), w)

def save_images(imgs, log_dir):
  """
  imgs: np.array in shape (n_imgs, height, width)
  log_dir: str
  """
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  
  for idx, img in enumerate(imgs):
    f_name = os.path.join(log_dir, u"img"+six.text_type(idx)+u".png")
    imsave(f_name, np.uint8(255*img))

def xentropy(data, reconst):
  ents = data * safe_log(reconst) + (1-data) * safe_log(1 - reconst)
  return - np.mean(np.sum(ents, axis=1))

def inner_prod(b1, c1, w1, b2, c2, w2):
  return (np.dot(b1, b2) + np.dot(c1, c2) +
          np.dot(w1.flatten(), w2.flatten()))

def safe_log(x):
  x = np.clip(x, LOG_MINIMUM, np.inf)
  return np.log(x)
