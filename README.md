# Music Genre Classification
>author: Carlos Lordelo

Build a machine learning (ML) model that takes audio files as input and returns a corresponding music genre.


## Description 
This repository contains jupyter notebooks related to the task of Music Genre Classification. We are going to use GTZAN dataset, which can be downloaded [here](http://opihi.cs.uvic.ca/sound/genres.tar.gz)

There are two jupyter notebooks in the repo that are self explanatory. You should start with `prepareDataAndFeatures.ipynb`, it will give you an overview of what should be done first when starting any machine learning project, including error verification, visualisation and preprocessing. This notebook will also show a set of handcrafted features we can compute from the music signals and use to build a genre classifier. I have already computed those features for the GTZAN and included on `./dataset/features.csv`.

Then, you should read `trainModels.ipynb`. This notebook discusses principal compoonent analysis (PCA), cross validation and hyperparameter search. It will show you how we can train and evaluate traditional machine learning classifiers for performing music genre classification on music signals, as well as, deep learning models. I have also included a novel architecture for genre classification adapted from a previous paper in which I proposed a state-of-the-art instrument recogniser. Here is the original work
>**C. Lordelo**, E. Benetos, S. Dixon and S. Ahlbäck, "Pitch-Informed Instrument Assignment Using a Deep Convolutional Network with Multiple Kernel Shapes" in *International Society for Music Information Retrieval Conference (ISMIR)*, October 2021. &nbsp; [<sub><sup>[pdf]</sup></sub>](https://archives.ismir.net/ismir2021/paper/000048.pdf) 

It should be really easy to follow along by just reading the notebooks. If you don't have a GPU to retrain the models that I show on the notebooks, don't worry, I have also included pre-trained models on `./callbacks/*` directory. Feel free to use them. 

I hope you like it. Let me know if you have any suggestions.

## Installation Instructions
In order to be able to run the notebooks you will need to have `python3` installed on your machine. You can them install the requirement packages by running 
```
pip install -r requirements.txt
```
## Conda instructions
If you are running conda you can setup a fresh environment and install the required packages by running:

```
conda create -n genre-classif-env pip
conda activate genre-classif-env
pip install -r requirements.txt
```