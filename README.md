# Detection by Sampling: Massive MIMO Detector based on Langevin Dynamics

Massive MIMO detector based on an annealed version of the Unadjusted Langevin Algorithm (ULA)

This repo contains the official implementation of the "Detection by Sampling: Massive MIMO Detector based on Langevin Dynamics" paper (arXiv:2202.12199). Link: 

## Code

The code is written in Python3 and run using PyTorch. Details as follows.

Dependencies

The following Python libraries are required: os, numpy, matplotlib, pickle, scipy, torch, pathlib

## Data

Data is compress in a zip file. In order to run the following methods, please decompress it in the folder. Otherwise, you can generate your own channels. 

### Concept

The main file is ```main.py```. This file call two runners: langevin runnner ```runner_langevin.py``` and ```runner_classical_methods.py```, which call the corresponding detector. Those detectos are in the folder ```model``` as well as the learning-based. To run the learning-based, there are a notebook for each method (both methods where taken from the repository https://github.com/krpratik/RE-MIMO). Then in the folder ```data``` you have the file with the chanels used in the paper as well as the ```sample_generator.py``` class. Then in ```utils``` there are two scripts: one with utility functions and the other to plot the results. Finally, in the folder ```results``` will be stored the output data (Symbol error rate).

### Run

To generate the plot in the paper (comparing the method), please run the script ```results_paper.py```. This will generate data for the classical detector (MMSE and SDR) and the Langevin detector. If you want to add the learning-based, in model there are two notebooks where you can train the model and test in the data channel.


### Remark

Sample_generator file and classic_detector are largely based on the code from: https://github.com/krpratik/RE-MIMO ("RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection")

## Feedback

For questions and comments, feel free to send me an email nzilberstein@rice.edu.

[1] Chowdhury A, Verma G, Rao C, Swami A, Segarra S. ML-aided power allocation for Tactical MIMO. 
arXiv preprint arXiv:2109.06992 2021 Sep 14.

BibTeX format:

@article{chowdhury2021ml,
  title={ML-aided power allocation for Tactical MIMO},
  author={Chowdhury, Arindam and Verma, Gunjan and Rao, Chirag and Swami, Ananthram and Segarra, Santiago},
  journal={arXiv e-prints},
  pages={arXiv--2109},
  year={2021}
}




## Citation

BibTeX format:

```
@article{zilberstein2022detection,
      title={Detection by Sampling: Massive MIMO Detector based on Langevin Dynamics}, 
      author={Nicolas Zilberstein and Chris Dick and Rahman Doost-Mohammady and Ashutosh Sabharwal and Santiago Segarra},
      year={2022},
      eprint={2202.12199},
      archivePrefix={arXiv},
}
```
