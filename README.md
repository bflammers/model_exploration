# Model exploration
This repository contains code for exploring various anomaly detection models as well as comparing their performance on realistic data sets. 

The content of the data/ folder is added to the .gitignore and will thus not be pushed to the remote repo on Github, with the exception of the data/datasets.json file. This json file contains download locations and other information about anomaly detection datasets.

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
And then add a new line with the package name to the environment.yml file using a text editor or from the command line with `echo "  - <package-name>" >> environment.yml`.

***
<br/>

## Incrementally install new libraries

If you pull from Github, and a new library is used in the code, you will get an error similar to:

> ModuleNotFoundError: No module named 'h5py'

This is because the new library has not been installed in your local model_exploration conda environment. Assuming the new package has been added to the environment.yml file, you can incrementally install new packages with the following command:

```
conda env update -n model_exploration -f environment.yml
```




