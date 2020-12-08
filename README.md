# Glow-MXNet

This repository contains an implementation for Glow in Apache MXNet.

```bash
| Overview ────────────────────────────────────────────────────────────|
|
├── train_mnist.py :: main training file
├── glow.py :: glow top level architecture 
├── layers.py :: building blocks of the flow network
├── utils.py :: utilities to train the network
├── config.py :: configuration file that defines the training procedure
└── .
```

## Usage

To run and train on MNIST samples, simply modify the parameters in `config.py` and execute:

> python train_mnist.py

## References

```
@inproceedings{kingma2018glow,
  title={Glow: Generative flow with invertible 1x1 convolutions},
  author={Kingma, Durk P and Dhariwal, Prafulla},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10215--10224},
  year={2018}
}
```
