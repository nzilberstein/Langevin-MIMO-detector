# Annealed Langevin Dynamics for Massive MIMO Detection

Massive MIMO detector based on an annealed version of the annealed Unadjusted Langevin Algorithm (ULA).

This repo contains the official implementation of the "Annealed Langevin Dynamics for Massive MIMO Detection" paper (arXiv.2205.05776, link: https://arxiv.org/abs/2205.05776) and "Detection by Sampling: Massive MIMO Detector based on Langevin Dynamics" paper (arXiv:2202.12199, link: https://arxiv.org/abs/2202.12199)

## Code

The code is written in Python3 and run using PyTorch. Details as follows.

Dependencies

The following Python libraries are required: <strong>os, numpy, matplotlib, pickle, scipy, torch, pathlib, gurobi package</strong>

### PreTrained models and channels dataset

https://drive.google.com/drive/folders/1cyUo3G_6d_Gc81ug1G_qcGIsRw4Lf2Zn?usp=sharing

To replicate the results, the batch of channels used in each experiment is given. They are in the link given. In order to run the different experiments, please copy them to the folder data. 

Otherwise, if you want to generate a new batch, or even use your own channel model, you can generate your own channels. For genertate channel model, just chane the flag loadChannelInput. For new model, you will need to add a line to load from the corresponding file.

### Concept

To facilitate the reproducibility, in the folder ```notebooks.py``` you can find a notebook for each experiment. For the non-learning based methods, you just run the notebooks and you can generate the plots. In a nutshell, each notebook call some runners, that are in charge on running each detector. 

The **notebooks** are:

      1. Results_synthetic_data: This notebook generate (the non-learning based) results of Fig. 4(b). To generate Fig. 4(a), Figs. 4(c) and (d) and Fig. 5(a), just you need to change the input file (download the channel dataset and uncomment the corresponding lines) and the config values to the corresponding setting.

      2. Results_ML: This notebook generate the results for the ML. It is a separate notebook because Gurobi is necessary, and not everyone has access to it. 

      3. Results_real_data and Results_real_data_symbol: These notebooks generate the results for real data. The first notebook go through the entire dataset, while the second one pick only one time frame (it generates Figs. 5(c) and (d)).

      4. MMNetiid, RE-MIMO, OAMPNet: These notebooks are the implementations. OAMPNet is design with the same style as this repo, while RE-MIMO is adapted from the repository. Therefore, in the notebook you will find some functions (like the train function). Notice also that there is a 3gpp notebook for OAMPNet and RE-MIMO also.

      5. Results_two_modulations: This notebook generates Fig. 6(a)

      5. Langevin_learinig_final: This notebook is the implementation of U-Langevin.

Notice that there are some runners based on the experiment. This is mainly because the pre-process for some dataset is different.

For the learning-based method, you can load the trained model given, or you will need to train the models. 

In the folder ```model``` you can find all the implementation: classical detectors, Langevin detector, learning-based as well as the MLP.

To generate data, the class defined in ```sample_generator.py``` handle the generation of observations, channel as well as the symbols. 

In ```utils``` there are two files: one with utility functions and the other to plot the results. 

Finally, in the folder ```results``` will be stored the output data (Symbol error rate).

Notice that is also a file main, which allows to run the detector by command line.

### Run

To generate the plot in the paper (comparing the method), please run the notebooks as it was explained. 

### Remark

*Sample_generator file, classic_detector and both learning-based baseline methods are largely based on the code from: https://github.com/krpratik/RE-MIMO ("RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection").

## Feedback

For questions and comments, feel free to send me an email nzilberstein@rice.edu.

## Citation
BibTeX format for journal paper:

```
@article{zilberstein2022detection_journal,
      title={Annealed Langevin Dynamics for Massive MIMO Detection}, 
      author={Nicolas Zilberstein and Chris Dick and Rahman Doost-Mohammady and Ashutosh Sabharwal and Santiago Segarra},
      year={2022},
      eprint={2205.05776},
      archivePrefix={arXiv}
}
```

BibTeX format for conference paper:

```
@article{zilberstein2022detection,
      title={Detection by Sampling: Massive MIMO Detector based on Langevin Dynamics}, 
      author={Nicolas Zilberstein and Chris Dick and Rahman Doost-Mohammady and Ashutosh Sabharwal and Santiago Segarra},
      year={2022},
      eprint={2202.12199},
      archivePrefix={arXiv}
}
```
