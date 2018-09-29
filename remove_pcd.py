from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import six
import shutil
import argparse
import numpy as np
from scipy.special import expit
from matplotlib import pylab as plt

from utils import xentropy, load_mnist, load_params, save_images, save_params
from rbm_utils import (calc_kl_grad, calc_c_grad,
    reconst, gibbs_samp, gibbs_samp_per, calc_remove_cost_app, temp_trans)

def main():
  data = load_mnist(FLAGS.input_path)
  train_idx = np.random.permutation(data.shape[0])
  train = data[train_idx[:60000]]
  test = data[train_idx[60000:]]
  
  b, c, w = load_params(FLAGS.param_root)
  
  base_name = os.path.basename(FLAGS.param_root)
  log_dir = os.path.join(
      FLAGS.log_root, "alg3-" + "n_hid%d-"%len(c) + base_name)
  if os.path.exists(log_dir): shutil.rmtree(log_dir)
  os.makedirs(log_dir)
  
  fout_name = os.path.join(log_dir, "record.txt")
  fout = io.open(fout_name, "w", encoding="utf-8")
  fout.write(u"epoch"); fout.write(u"\t")
  fout.write(u"target_node_idx"); fout.write(u"\t")
  fout.write(u"remove_cost"); fout.write(u"\t")
  fout.write(u"n_remain"); fout.write(u"\t")
  fout.write(u"reconst_cost"); fout.write(u"\t")
  fout.write(u"\n"); fout.flush()

  img_idxs = np.random.choice(len(train), FLAGS.batch_size)
  batch = train[img_idxs]
  s_vis, s_hid = temp_trans(b, c, w, batch, 100, 1e-3)
  
  epoch = 0
  while True:
    while True:
      epoch += 1
      # Calculate remove cost
      img_idxs = np.random.choice(len(train), FLAGS.batch_size)
      batch = train[img_idxs]
      s_vis, s_hid = gibbs_samp_per(b, c, w, s_vis, FLAGS.n_gibbs_samp)
      
      s_vis_f = np.float32(s_vis)
      s_hid_f = np.float32(s_hid)
      
      remove_cost, rc_std = calc_remove_cost_app(
          b, c, w, batch, s_vis_f, s_hid_f, FLAGS.num_geo)
        
      # Loop to lower the remove cost
      target = np.argmin(remove_cost)
      rc = remove_cost[target]
      std = rc_std[target]
        
      if rc + FLAGS.mul_std * std < FLAGS.cut_thre:
        c = np.delete(c, target)
        w = np.delete(w, target, axis=1)
        s_vis, s_hid = temp_trans(b, c, w, s_vis, 100, 1e-3)
        # Logging
        fout.write(six.text_type(epoch)); fout.write(u"\t")
        fout.write(six.text_type(target)); fout.write(u"\t")
        fout.write(six.text_type(rc)); fout.write(u"\t")
        fout.write(six.text_type(len(c))); fout.write(u"\t")
        cost = xentropy(batch, reconst(b, c, w, batch))
        fout.write(six.text_type(cost)); fout.write(u"\t")
        fout.write(u"\n"); fout.flush()
      
        if epoch % FLAGS.save_per == 0:
          img_log_dir = os.path.join(log_dir, "gen_imgs-epoch%d"%epoch)
          os.makedirs(img_log_dir)
          img_idxs = np.random.choice(len(data), FLAGS.n_save_imgs)
          imgs = reconst(b, c, w, data[img_idxs])
          save_images(imgs.reshape(-1, 28, 28), img_log_dir)
        
          params_log_dir = os.path.join(log_dir, "params-epoch%d"%epoch)
          os.makedirs(params_log_dir)
          save_params(params_log_dir, b, c, w)
        
      else:
        break
      
    cd_b, cd_c, cd_w, std_db, std_dc, std_dw = calc_kl_grad(
          b, c, w, batch, s_vis_f, s_hid_f)
    
    rc, rc_b, rc_c, rc_w, std_cb, std_cc, std_cw = calc_c_grad(
          b, c, w, batch, s_vis_f, s_hid_f, target)
  
    remove_cost, rc_std = calc_remove_cost_app(
      b, c, w, batch, s_vis_f, s_hid_f, FLAGS.num_geo)
    
    rc = remove_cost[target]
    std = rc_std[target]

    mask_b = (cd_b * rc_b / std_db / std_cb)
    mask_c = (cd_c * rc_c / std_dc / std_cc)
    mask_w = (cd_w * rc_w / std_dw / std_cw)
    
    mask_b[np.isnan(mask_b)] = np.inf
    mask_c[np.isnan(mask_c)] = np.inf
    mask_w[np.isnan(mask_w)] = np.inf
    
    mask_b = np.random.binomial(1, expit(mask_b))
    mask_c = np.random.binomial(1, expit(mask_c))
    mask_w = np.random.binomial(1, expit(mask_w))
    
    if np.sum(mask_b) + np.sum(mask_c) + np.sum(mask_w) == 0:
      break
      
    # Update paramters
    b -= FLAGS.move_rate * cd_b * mask_b
    c -= FLAGS.move_rate * cd_c * mask_c
    w -= FLAGS.move_rate * cd_w * mask_w
    
    # Logging
    fout.write(six.text_type(epoch)); fout.write(u"\t")
    fout.write(six.text_type(target)); fout.write(u"\t")
    fout.write(six.text_type(rc)); fout.write(u"\t")
    fout.write(six.text_type(len(c))); fout.write(u"\t")
    cost = xentropy(batch, reconst(b, c, w, batch))
    fout.write(six.text_type(cost)); fout.write(u"\t")
    fout.write(u"\n"); fout.flush()
      
    if epoch % FLAGS.save_per == 0:
      img_log_dir = os.path.join(log_dir, "gen_imgs-epoch%d"%epoch)
      os.makedirs(img_log_dir)
      img_idxs = np.random.choice(len(data), FLAGS.n_save_imgs)
      imgs = reconst(b, c, w, data[img_idxs])
      save_images(imgs.reshape(-1, 28, 28), img_log_dir)
    
      params_log_dir = os.path.join(log_dir, "params-epoch%d"%epoch)
      os.makedirs(params_log_dir)
      save_params(params_log_dir, b, c, w)

if __name__ == "__main__":
  parser = argparse.ArgumentParser("alg3")
  parser.add_argument("--param_root", type=str, default="trained_params")
  parser.add_argument("--input_path", type=str, default="input/mnist.pkl.gz")
  parser.add_argument("--log_root", type=str, default="logs")
  parser.add_argument("--n_save_imgs", type=int, default=10)
  parser.add_argument("--save_per", type=int, default=5000)
  parser.add_argument("--batch_size", type=int, default=int(1e3))
  parser.add_argument("--n_gibbs_samp", type=int, default=1)
  parser.add_argument("--cut_thre", type=float, default=0.0)
  parser.add_argument("--move_rate", type=float, default=0.01)
  parser.add_argument("--num_geo", type=int, default=1)
  parser.add_argument("--mul_std", type=int, default=1)
  FLAGS = parser.parse_args()
  
  main()
