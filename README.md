# Model exploration
This repository contains code for exploring various anomaly detection models as well as testing and comparing their performance on realistic data sets. 

The data/ folder is added to the .gitignore and will thus not be pushed to the remote repo on Github.

***
<br/>

## Getting started

After cloning this repository using `git clone git@github.com:bflammers/model_exploration.git`, or using the Git desktop client, we need to install the required python libraries. All the libraries are defined in the environment.yml file. If you do not have python on your machine yet, I recommend installing one of the newer versions (>=3.7) with Anaconda. 

Create an environment on your local machine using the environment.yml file. Using Anancoda: 
```
conda env create -f environment.yml
```

Then activate the environment:
```
conda activate model_exploration
```

And start the jupyter notebook server:
```
jupyter notebook
```

If your conda environment is not found by jupyter notebook, try installing nb_conda in your base environment:
```
conda deactivate
conda install nb_conda
```

***
<br/>

## Adding new code

For each new model to explore, we add a new jupyter notebook in the notebooks/ folder so we can interactively make plots and quickly get a feel for a model. Helper code can be place in .py files in the src/ folder, this allows to nicely import the code in other scripts. 

If you need a new package, but it is not currently installed in the environment, install it in the environment using:
```
conda install -n model_exploration <package-name>
```
And then add a new line with the package name to the environment.yml file. 





