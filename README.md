Decreasing the size of the RBM
----
An implementation of the paper[1] in which the authors proposed a method to sustain the performance of an RBM while decreasing the number of hidden nodes.

## Run demo
```
pip install -r requirements.txt
python remove_pcd.py
```
The script starts removing the hidden nodes of an RBM whose parameters are given in folder `trained_params`.

The result of the simulation will appear in `RBM/logs/` automatically.

## References
1. Y. Saito et al. (2018), Decreasing the size of the Restricted Boltzmann Machine, https://arxiv.org/abs/1807.02999