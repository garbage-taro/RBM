from __future__ import division
from __future__ import print_function

import os
import io
import glob
import six
import gzip
import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt
from tracker import py_lpf_det

def softplus(x):
  return np.maximum(0, x) + np.log(1+np.exp(-np.abs(x)))

def load_mnist(path):
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

def load_params(path):
  b = np.load(os.path.join(path, "b.npy"))
  c = np.load(os.path.join(path, "c.npy"))
  w = np.load(os.path.join(path, "w.npy"))
  return b, c, w

def reconst(data, b, c, w):
  m_hid = expit(np.dot(data, w) + c)
  m_vis = expit(np.dot(m_hid, w.T) + b)
  return m_vis

def calc_xentropy(data, b, c, w):
  m_vis = reconst(data, b, c, w)
  xentropy = - np.mean(np.sum(data * np.log(m_vis) +
	    (1-data)*np.log(1-m_vis), axis=1))
  return xentropy

def fe(data, b, c, w):
  val = - np.dot(data, b)
  val -= np.sum(softplus(np.dot(data, w) + c), axis=1)
  return val

def calc_kl(data, b, c, w):
  _, cnts = np.unique(data, return_counts=True, axis=0)
  cnts = np.float32(cnts)
  freqs = cnts / np.sum(cnts)
  lpf = py_lpf_det(b, c, w)
  kl = np.sum(freqs * np.log(freqs))
  kl += np.mean(fe(data, b, c, w))
  kl += lpf

  return kl

def main():
    
  fout = io.open("KLD.txt", "a", encoding="utf-8")
  fout.write(u"epoch"); fout.write(u"\t")
  fout.write(u"cross entropy"); fout.write(u"\t")
  fout.write(u"KLD"); fout.write(u"\t")
  fout.write(u"std"); fout.write(u"\t")
  fout.write(u"\n"); fout.flush()    

  imgs = load_mnist(MNIST_PATH)
  imgs = (imgs > 0.5).astype(np.bool)
  
  param_dirs = glob.glob(os.path.join(TGT_DIR, "params-epoch*"))
  n_params = len(param_dirs)
  epochs = np.empty(n_params, dtype=np.int)
  kls = np.empty(n_params, dtype=np.float)
  std_kls = np.empty(n_params, dtype=np.float)
  xents = np.empty(n_params, dtype=np.float)
  for idx, param_dir in enumerate(param_dirs):
    
    epoch_str = os.path.basename(param_dir)[len("params-epoch"):]
    epochs[idx] = int(epoch_str)
    
    print("Start calculation for epoch", epochs[idx])
    b, c, w = load_params(param_dir)
    xents[idx] = calc_xentropy(imgs, b, c, w)
    print("---", "cross entropy was", xents[idx])
    kls[idx], std_kls[idx] = calc_kl(imgs, b, c, w)
    print("---", "kl was", kls[idx], "pm", std_kls[idx])
  
    # Logging
    fout.write(six.text_type(epochs[idx])); fout.write(u"\t")
    fout.write(six.text_type(xents[idx])); fout.write(u"\t")
    fout.write(six.text_type(kls[idx])); fout.write(u"\t")
    fout.write(six.text_type(std_kls[idx])); fout.write(u"\t")
    fout.write(u"\n"); fout.flush()
  
  fout.close()
  srtd_idxs = np.argsort(epochs)
  epochs = epochs[srtd_idxs]
  xents = xents[srtd_idxs]
  kls = kls[srtd_idxs]
  std_kls = std_kls[srtd_idxs]
  
  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
  ax1.plot(epochs, xents)
  ax1.set_ylabel("Cross Entropy")
  ax2.plot(epochs, kls)
  ax2.errorbar(epochs, kls, yerr=std_kls)
  ax2.set_ylabel("KL divergence")
  ax2.set_xlabel("Epochs")
  plt.savefig(SAVE_PATH)

if __name__ == "__main__":
  MNIST_PATH = "input/mnist.pkl.gz"
  TGT_DIR = "pcd_500"
  SAVE_PATH = "comp_kld_reconst_pcd_500.png"
  main()
