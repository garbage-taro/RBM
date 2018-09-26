from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import expit
from utils import safe_log
from datetime import datetime as dt

def reconst(b, c, w, data):
  m_h = expit(np.dot(data, w) + c)
  return expit(np.dot(w, m_h.T).T + b)

def gibbs_samp(b, c, w, batch, n_samp):
  s_vis = batch
  for i in range(n_samp):
    sm_hid = expit( np.dot(s_vis, w) + c )
    s_hid = np.random.binomial(1, sm_hid)
    sm_vis = expit( np.dot(w, s_hid.T).T + b )
    s_vis = np.random.binomial(1, sm_vis)
  return np.float32(s_vis), np.float32(s_hid)

def gibbs_samp_per(b, c, w, s_vis, n_samp):
  for i in range(n_samp):
    sm_hid = expit( np.dot(s_vis, w) + c )
    s_hid = np.random.binomial(1, sm_hid)
    sm_vis = expit( np.dot(w, s_hid.T).T + b )
    s_vis = np.random.binomial(1, sm_vis)
  return s_vis, s_hid

def temp_trans(b, c, w, s_vis, n_step, delta_beta):
  sm_hid = expit( np.dot(s_vis, w) + c )
  s_hid = np.random.binomial(1, sm_hid)
  s_vis_test = s_vis.copy() 
  s_hid_test = s_hid.copy()
  log_prob = energy(delta_beta * b, delta_beta * c, delta_beta * w, 
                      s_vis, s_hid)

  # Melting process
  for i in range(1, n_step):
    beta = 1.0 - delta_beta * i
    sm_vis = expit( np.dot(beta * w, s_hid_test.T).T + beta * b )
    s_vis_test = np.random.binomial(1, sm_vis)
    sm_hid = expit( np.dot(s_vis_test, beta * w) + beta * c )
    s_hid_test = np.random.binomial(1, sm_hid)    

    log_prob += energy(delta_beta * b, delta_beta * c, delta_beta * w, 
                        s_vis_test, s_hid_test)

  beta = 1.0 - delta_beta * n_step
  sm_vis = expit( np.dot(beta * w, s_hid_test.T).T + beta * b )
  s_vis_test = np.random.binomial(1, sm_vis)
  sm_hid = expit( np.dot(s_vis_test, beta * w) + beta * c )
  s_hid_test = np.random.binomial(1, sm_hid)  
  
  # Annealing process
  for i in range(1, n_step, -1):
    beta = 1.0 - delta_beta * i
    sm_vis = expit( np.dot(beta * w, s_hid_test.T).T + beta * b )
    s_vis_test = np.random.binomial(1, sm_vis)
    sm_hid = expit( np.dot(s_vis_test, beta * w) + beta * c )
    s_hid_test = np.random.binomial(1, sm_hid) 

    log_prob -= energy(delta_beta * b, delta_beta * c, delta_beta * w, 
                        s_vis_test, s_hid_test)
  prob = np.minimum(1, np.exp(log_prob))
  # print("mean transition probability is %f" %np.mean(prob))
  bool_trans = np.random.uniform(0, 1, s_vis.shape[0]) < prob
  
  s_vis = ( s_vis_test.T * bool_trans + s_vis.T * (1 - bool_trans) ).T
  s_hid = ( s_hid_test.T * bool_trans + s_hid.T * (1 - bool_trans) ).T
    
  return np.float32(s_vis), np.float32(s_hid) 

def energy(b, c, w, s_vis, s_hid):
  return - np.dot(b, s_vis.T) - np.dot(c, s_hid.T) - np.sum(np.dot(s_vis, w) * s_hid, axis=1)

def calc_remove_cost_app(b, c, w, data, s_vis, s_hid, n_terms):
  m_hid = expit( np.dot(data, w) + c )
  remove_cost_1st = -np.mean(safe_log(1.0 - m_hid), axis=0)
  
  ag = np.mean(s_hid, axis=0)
  geom = [np.sum(ag ** n / n for n in range(1, n_terms + 1))]
  remove_cost_2nd = np.array(geom).reshape(len(c))
  
  remove_cost = remove_cost_1st - remove_cost_2nd
  
  mu1 = np.mean(np.log(1.0 - m_hid), axis=0)
  var1 = np.mean(np.log(1.0 - m_hid)**2, axis=0) - mu1**2
  var1 /= (data.shape[0] - 1.0)

  mu2 = np.mean(s_hid, axis=0)
  var2 = mu2 * (1.0 - mu2) / (s_hid.shape[0] - 1.0)
  
  std = np.sqrt(var1 + var2)
  
  return remove_cost, std

def calc_remove_cost(b, c, w, data, s_vis, s_hid):
  m_hid = expit( np.dot(data, w) + c )
  remove_cost = (-np.mean(safe_log(1.0 - m_hid), axis=0) +
      safe_log(np.mean(1.0 - s_hid, axis=0)))
  
  mu1 = np.mean(np.log(1.0 - m_hid), axis=0)
  var1 = np.mean(np.log(1.0 - m_hid)**2, axis=0) - mu1**2
  var1 /= data.shape[0]

  mu2 = np.mean(1.0 - s_hid)
  var2 = (1.0 - mu2) / (mu2 * s_hid.shape[0])
  
  std = np.sqrt(var1 + var2)
  
  return remove_cost, std

def calc_kl_grad_no_var(b, c, w, batch, s_vis, s_hid):    
  m_hid = expit(np.dot(batch, w) + c)
    
  cd_b = - np.mean(batch, axis=0) + np.mean(s_vis, axis=0)
  cd_c = - np.mean(m_hid, axis=0) + np.mean(s_hid, axis=0)
  cd_w = (- np.dot(batch.T, m_hid) / len(batch) +
            np.dot(s_vis.T, s_hid) / len(s_vis))
    
  return cd_b, cd_c, cd_w

def calc_kl_grad(b, c, w, batch, s_vis, s_hid):    
  m_hid = expit(np.dot(batch, w) + c)

  mean_v_1st = np.mean(batch, axis=0)
  mean_h_1st = np.mean(m_hid, axis=0)
  mean_w_1st = (np.dot(batch.T, m_hid) / len(batch)).flatten()
    
  mean_v_2nd = np.mean(s_vis, axis=0)
  mean_h_2nd = np.mean(s_hid, axis=0)
  mean_w_2nd = (np.dot(s_vis.T, s_hid) / len(s_vis)).flatten()

  # evaluation of gradient
  cd_b = - mean_v_1st + mean_v_2nd
  cd_c = - mean_h_1st + mean_h_2nd
  cd_w = (- mean_w_1st + mean_w_2nd).reshape(w.shape[0], w.shape[1])

  # evaluation of std
  var_b = (mean_v_1st - mean_v_1st * mean_v_1st) / (batch.shape[0] - 1.0) \
          + (mean_v_2nd - mean_v_2nd * mean_v_2nd) / (s_vis.shape[0] - 1.0)

  var_c = (np.mean(m_hid * m_hid, axis=0) - mean_h_1st * mean_h_1st) / (batch.shape[0] - 1.0) \
          + (mean_h_2nd - mean_h_2nd * mean_h_2nd) / (s_hid.shape[0] - 1.0)

  var_w = ((np.dot(batch.T, m_hid * m_hid).flatten()) / batch.shape[0] 
              - mean_w_1st * mean_w_1st) / (batch.shape[0] - 1.0) \
          + (mean_w_2nd - mean_w_2nd * mean_w_2nd) / (s_hid.shape[0] - 1.0)
  
  var_b = np.sqrt(var_b) 
  var_c = np.sqrt(var_c) 
  var_w = np.sqrt(var_w.reshape(w.shape[0], w.shape[1])) 

  return cd_b, cd_c, cd_w, var_b, var_c, var_w

def calc_c_grad(b, c, w, batch, s_vis, s_hid, target):
  m_hid = expit(np.dot(batch, w) + c)
  s_vis_bar = s_vis[s_hid[:, target] == 0]
  s_hid_bar = s_hid[s_hid[:, target] == 0]

  mean_v_1st = np.mean(s_vis_bar, axis=0)
  mean_h_1st = np.mean(s_hid_bar, axis=0)
  mean_w_1st = (np.dot(s_vis_bar.T, s_hid_bar) / 
                  s_vis_bar.shape[0]).flatten()
  
  mean_v_2nd = np.mean(s_vis, axis=0)
  mean_h_2nd = np.mean(s_hid, axis=0)
  mean_w_2nd = (np.dot(s_vis.T, s_hid) / s_vis.shape[0]).flatten()
  
  mean_c_3rd = np.mean(m_hid[:, target])
  mean_w_3rd = (np.dot(m_hid[:, target], batch) / len(batch))
 
  # Evaluate remove cost
  rc = (-np.mean(safe_log(1.0 - m_hid[:, target])) +
          safe_log( 1.0 - np.mean(s_hid[:, target], axis=0)))
  
  # Evaluate gradient
  rc_b = mean_v_1st - mean_v_2nd
  rc_c = mean_h_1st - mean_h_2nd
  rc_w = (mean_w_1st - mean_w_2nd).reshape(w.shape[0], w.shape[1])
    
  rc_c[target] += mean_c_3rd
  rc_w[:, target] += mean_w_3rd
  
  # Evaluate std 
  var_b = (mean_v_1st - mean_v_1st * mean_v_1st) / (s_vis_bar.shape[0] - 1.0) \
          + (mean_v_2nd - mean_v_2nd * mean_v_2nd) / (s_vis.shape[0] - 1.0)

  var_c = (mean_h_1st - mean_h_1st * mean_h_1st) / (s_hid_bar.shape[0] - 1.0) \
          + (mean_h_2nd - mean_h_2nd * mean_h_2nd) / (s_hid.shape[0] - 1.0)
  
  var_c[target] += (np.mean(m_hid[:, target] * m_hid[:, target]) 
                  - mean_c_3rd * mean_c_3rd) / (len(m_hid) - 1.0)

  var_w = ((mean_w_1st - mean_w_1st * mean_w_1st) / (s_hid_bar.shape[0] - 1.0) \
          + (mean_w_2nd - mean_w_2nd * mean_w_2nd) / (s_hid.shape[0] - 1.0)).reshape(w.shape[0], w.shape[1])
            
  var_w[:, target] += (np.dot(m_hid[:, target] * m_hid[:, target], batch) / len(batch) \
                      - mean_w_3rd * mean_w_3rd) / (len(batch) - 1.0)
  
  var_b = np.sqrt(var_b) 
  var_c = np.sqrt(var_c) 
  var_w = np.sqrt(var_w) 
    
  return rc, rc_b, rc_c, rc_w, var_b, var_c, var_w

