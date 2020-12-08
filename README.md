# Glow-MXNet

This repository contains an implementation for Glow in Apache MXNet.

```bash
| Overview ────────────────────────────────────────────────────────────────────────────────|
|
├── train_mnist.py :: main training file
├── glow.py :: glow top level architecture 
├── layers.py :: building blocks of the flow network
├── utils.py :: utilities for training
├── config.py :: configuration file for training procedure
└── .
```

## Usage

To run and train on MNIST samples, simply modify the parameters in `config.py` and execute:

> python train_mnist.py
