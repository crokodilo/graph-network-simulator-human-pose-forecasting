# Graph Network-based Simulator (GNS) for Human-Pose-Forecasting
This repository contains an implementation of the paper [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405) to perform Human Pose Forecasting.
For more details about the model implementation click [here](https://github.com/crokodilo/Graph-Network-Simulator-Human-Pose-Forecasting/blob/main/paper.pdf)

## Install dependencies

```console
conda env create -f environment.yml
```

## Download dataset
[Human3.6M](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure:

```console
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

Put the all downloaded datasets in ../datasets directory.

## Acknowledgments
Some of our code was adapted from [STS-GCN](https://github.com/FraLuca/STSGCN), [Learning to Simulate](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) and [Graph Network Simulator (GNS)](https://github.com/geoelements/gns).
