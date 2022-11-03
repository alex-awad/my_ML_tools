# my_ML_tools
Python package that facilitates common operations in Machine Learning workflow
with "classical" Machine Learning models. Currently only works for regression 
problems. 

## Outline
This package comprises different modules: data_tools to import and pre-process datasets, plot_tools for 
exploratory data analysis and evaluation of the results form the trained models,
and model_tools for automatic training, tuning, and evaluation of different 
Machine Learning models.

Currently implemented models: 
- Lasso regression (scikit-learn)
- Ridge regression (scikit-learn)
- Random Forest regression (scikit-learn) 
- SVM regression (scikit-learn)
- GBM regression (XGBoost)
- Multilayer Perceptron regression (Keras)

## Installation

Currently, there are two ways to install this package.

### 1) Installation via pip

To install this package via pip, run the following command

```
pip install git+https://github.com/alex-awad/my_ML_tools
```

### 2) Installation from cloned repository

To install a modified version of this package, first clone this repository by running

```
git clone https://github.com/alex-awad/my_ML_tools.git
```

or 

```
git clone git@github.com:alex-awad/my_ML_tools.git
```

and then run the following command from the root of the cloned directory

```
pip install .
```




