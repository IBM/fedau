## Code for paper "A Lightweight Method for Tackling Unknown Participation Statistics in Federated Averaging"

The code was run successfully in the following environment: Python 3.9, PyTorch 1.13.1, Torchvision 0.14.1

See `config.py` for all the configurations. Some examples on reproducing the CIFAR-10 results (with 5 different random seeds) and Bernoulli participation are as follows.

```
python3 main.py -data cifar10 -weighting adaptive -iters-total 50000 -seeds 1,2,3,4,5 -lr 0.1

python3 main.py -data cifar10 -weighting adaptive -k-adaptive 50 -iters-total 50000 -seeds 1,2,3,4,5 -lr 0.1

python3 main.py -data cifar10 -weighting average_participating -iters-total 50000 -seeds 1,2,3,4,5 -lr 0.0562 -lr-global 1.78

python3 main.py -data cifar10 -weighting average_all -iters-total 50000 -seeds 1,2,3,4,5 -lr 0.1 -lr-global 10

python3 main.py -data cifar10 -weighting fedvarp -iters-total 50000 -seeds 1,2,3,4,5 -lr 0.0562 

python3 main.py -data cifar10 -weighting mifa -iters-total 50000 -seeds 1,2,3,4,5 -lr 0.0562 

python3 main.py -data cifar10 -weighting known_prob -iters-total 50000 -seeds 1,2,3,4,5 -lr 0.0562

```

By default, the results are saved in `results_*.csv`.

