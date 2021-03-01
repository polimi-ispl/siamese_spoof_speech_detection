# Synthetic Speech Detection using Siamese Network and VGGish

This repository contains the code for the task of spoof speech detection using Siamese structure with VGGish architecture. The system is trained and tested on  on [ASV spoof 2019](https://www.asvspoof.org/index2019.html) dataset.


In particular the code implements:
- input features computation, i.e. logmelspectrograms following the parameters for VGGish architecture;
- data generator definition for selecting couples of samples as input to the siamese network;
- model definition of the siamese configuration with VGGish;
- training and evaluation scripts on ASVspoof dataset;
