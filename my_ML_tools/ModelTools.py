""" Module to facilitate model training, optimization, and evaluation.
    Includes class ModelEvaluator (only static methods) and ModelTrainers.
    The different Modeltrainer classes inherit from Modeltrainer base class.
    Currently there is a Modeltrainer class that uses a training set and a test
    set (Modeltrainer_train_test) and a Modeltrainer class that uses the whole
    dataset as training set and LOO-cross-validation as a test set.

"""


PYTHONHASHSEED=0 # To get reproducible results (with Keras)

from random import Random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut

import tensorflow as tf
import keras

from .PlotTools import PlotTools
from .additional.KerasLayers import get_layer_list
from .custom_models import AveragePredictor

# Set seeds to get reproducible results
import random as python_random

np.random.seed(0)
python_random.seed(0)
tf.random.set_seed(0)


class ModelEvaluator:
    """ Class for creation of predictions and evaluation of trained models.
        Using mostly static methods.
    """

    def __init__(self):
        ...

    
    @staticmethod
    def RMSE(preds, targets):
        """Return root-mean-squared-error from predictions. """
        return mean_squared_error(preds, targets, squared=False)


    @staticmethod
    def MAE(preds, targets):
        """Return root-mean-absolute-error from predictions. """
        return mean_absolute_error(preds, targets)


    @staticmethod
    def top_weights_linear(
        model,
        train_inputs,
        num_top=7,
        model_name="SET MODEL NAME!",
        save_fig=False,
        figure_dir="",
        show_title=True,
        figsize=(20,15),
        font_scale=3
        ):
        """ Plots the highest and the lowest weights from trained linear models
            in a bar chart. 
        
        Parameters:
        ----------

        model: Trained linear model from scikit-learn.

        train_inputs (pandas.DataFrame): DataFrame used for model training. 

        num_top (int): Number of highest and lowest weights to include in the
            bar chart. Defaults to 7.

        model_name (String): Name of the model displayed in the title and for
            the name of the saved file. Defaults to "SET MODEL NAME!" to
            assert that the name is set. 

        save_fig (Boolean): Whether to save the figure. Defaults to False.

        figure_dir (String): Location of directory to save the figure in. 
            Defaults to "" (working directory).

        show_title (Boolean): Whether to display title in the plot. 
            Defaults to True.

        figsize (Tuple): Size of the figure (according to matplotlib.pyplot).
            Defaults to (20,15).

        font_scale (float): Scale of the font according to searborn.set_context.
            Defaults to 3.

        Returns:
        ---------

        top_weights_df (pandas.DataFrame): DataFrame containing the highest
            and lowest weights. 

        """
        # Create a list with the models' weights
        weights = model.coef_

        weights_df = ( pd.DataFrame({
            'columns': train_inputs.columns,
            'weight': weights
        }).sort_values('weight', ascending=False) )

        # Set plot appearance
        sns.set_context("notebook", font_scale=font_scale)
        plt.rcParams["figure.figsize"] = figsize


        # tGet highest and lowest weights
        top_weights_df = weights_df.head(num_top) \
            .append(weights_df.tail(num_top))


        # Set color dependent on positive or negative value
        colors = [
            "darkorange" if c >= 0 else "darkred"
            for c in top_weights_df["weight"]]

        # Display plot title
        if show_title:
            plt.title(f"{model_name} - Top {num_top} features")

        # Plot the data
        ax = sns.barplot(
            data=top_weights_df,
            x='columns',
            y='weight',
            palette=colors,
            alpha=0.8,
            order=top_weights_df["columns"])

        # Adjust plot style
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        ax.axhline(0, color="k", clip_on=False)
        sns.despine(top=True)
        ax.xaxis.set_ticks_position('none') 
        plt.xlabel("")
        plt.tight_layout()

        # Save figure
        if save_fig:
            plt.savefig(
                figure_dir + model_name + " - Top weights" + ".svg",
                format="svg",
                bbox_inches="tight"
                )

        plt.show()

        # Reset size of plot and font
        sns.set_context("notebook", font_scale=1.0)

        # Return dataframe containing top weights
        return top_weights_df
    
    
    @staticmethod
    def grid_to_txt(
        grid,
        save_location="",
        save_name="grid.txt"
    ):
        """ Uses trained grid from GridSearchCV or RandomizedSearchCV and saves
        used hyperparameters and best hyperparameters in a text file.
        
        Parameters:
        ----------

        grid: Trained grid from GridSearchCV or RandomizedSearchCV
            (scikit-learn).
            
        save_location: Directory to save the results in. Defaults to current
            directory.
            
        save_name: Name of the saved file, needs to have .txt as ending.
            Defaults to "grid.txt"
            
        """
        with open(save_location+save_name, "w") as f:
            # Write parameters included in the grid
            f.write("Used grid:\n")
            if isinstance(grid, sklearn.model_selection.RandomizedSearchCV):
                for key, value in grid.param_distributions.items():
                    f.write(f"\t[{key}]: {value}\n")
            elif isinstance(grid, sklearn.model_selection.GridSearchCV):
                for key, value in grid.param_grid.items():
                    f.write(f"\t[{key}]: {value}\n")
            else:
                raise TypeError("")
            
            
            f.write("\n-----------------\n\n")
            # Print best hyperparameters
            f.write("Best hyperparameters:\n")
            for key, value in grid.best_params_.items():
                f.write(f"\t[{key}]: {value}\n")
                
                

class ModelTrainer_base:
    """ Parent class of modeltrainer. Automatically train and tune 
        models.
        Currently supported models: 
        Regression: Lasso, Ridge, Elastic Net, Random Forest, 
        Multilayer Perceptron.

    Attributes
    ----------
    
    my_cv (int or sklearn.model_selection.LeaveOneOut): (class attribute)
        Cross-validation folds. Defaults to Leave-One-Out (scikit-learn).

    n_jobs (int): (class attribute) Number of parallel jobs running
        (-1 to run all possible jobs). WARNING: Sometimes setting n_jobs to -1
        could cause errors. 

    train_lim (tuple<float,float>): (class attribute) Lower and upper limit for
    plotting the training set. If it is set to None, the limits are determined
    automatically every time. Defaults to None.

    test_lim (tuple<float,float>): (class attribute) Lower and upper limit for
    plotting the test set.  If it is set to None, the limits are determined
    automatically every time. Defaults to None.

    train_inputs (pandas.DataFrame): Features of training set.

    train_targets (pandas.Series): Labels of training set.
    
    test_inputs (pandas.DataFrame): Features of test set. Defaults to None.
    
    test_targets (pandas.Series): Labels of test set. Defaults to None.

    is_log (boolean): Set to true, if the label of dataset has been 
    transformed to log(10). Defaults to False.
    
    rs (int): Random-state for methods that use randomness.
        Defaults to 0.
        
    label_name (String): Name of the label. Automatically taken from
        train_targets.
        
    label_display (String): Name to display the labels in the parity plots.
            Defaults to label_name.
    """
    ## Class attributes
    my_cv = LeaveOneOut()
    n_jobs = -1
    train_lim = None
    test_lim =  None


    def __init__(
        self,
        train_inputs,
        train_targets,
        test_inputs=None,
        test_targets=None,
        is_log=False,
        rs=0,
        label_name=None,
        label_display=None
        ):
        
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.is_log = is_log
        self.rs = rs
        
        # Get name of the label
        if (label_name is None):
            self.label_name = train_targets.name
        else:
            self.label_name = label_name
        
        # Set name of displayed label name in plots
        if (label_display is None):
            self.label_display = self.label_name
        else:
            self.label_display = label_display


    ## Class methods

    @classmethod
    def set_train_lim(cls, lim):
        cls.train_lim = lim

    
    @classmethod
    def set_test_lim(cls, lim):
        cls.test_lim = lim


    ## Methods
    def train_and_evaluate(
        self,
        model,
        fig_title="",
        font_scale=1.5,
        save_fig=False,
        save_location="",
        save_name="Figure",
        save_grid_results=False,
        verbose=2
    ):
        """ Needs to be adjusted for child classes!"""
        raise(NotImplementedError("""This method needs to be implemented in 
            child class!"""))
    
    
    def evaluate(
        self,
        model,
        fig_title="",
        font_scale=1.5,
        save_fig=False,
        save_grid_results=False,
        save_location="",
        save_name="Figure"
    ):
        """ Needs to be adjusted for child classes!"""
        raise(NotImplementedError("""This method needs to be implemented in 
            child class!"""))
        

    ## Lasso Regression
    def tune_and_plot_lasso(
        self,
        start_alphas,
        iters_second=1000,
        save_fig=False,
        save_grid_results=False,
        save_location="",
        save_name="Lasso",
        verbose=2
        ):
        """ Optimizes the hyperparameter alpha from lasso regression.
            The optimal value of alpha is determined in two steps. Plots 
            curves with cross-validation RMSE of the optimizations and plots
            the parity plots of predictions from the trained models. 

        Parameters:
        ---------

        start_alphas (List<int>): Range with alphas to start the search with.

        save_fig (Boolean): Whether to save the figure. Defaults to False.
        
        save_grid_results (Boolen): Whether to save results of hyperparameter
            tuning as text file. Defaults to False.

        iters_second (int): Number of points for second optimization.
            Defaults to 1000.

        save_location (String): Directory to save the figures. Defaults to
            current directory

        save_name (String): Name of the saved figure. Defaults to "Figure".
        
        verbose (int): Verbosity of the output: 
            2: Print results and show parity plot.
            1: Show parity plot.
            0: Do not print and do not show parity plot. 
            Defaults to 2.


        Returns: 
        ---------

        tuple(grid (sklearn.model_selection.GridSearchCV),
            tuple(results, metrics)):
            grid: Results of hyperparameter optimization. The optimized models
            and metrics for the optimization can be obtained from this grid.
             
            (results, metrics): See return of self.train_and_evaluate().

        """

        model = Lasso()

        ## First optimization
        print("Optimizing Lasso regression model:")
        param_grid = dict(alpha=start_alphas)
        grid = GridSearchCV(
            model,
            param_grid,
            cv=self.my_cv,
            scoring='neg_root_mean_squared_error',
            n_jobs = self.n_jobs) # Use grid-search with cross-validation
        grid.fit(self.train_inputs, self.train_targets)

        # Get results from grid search
        mean_score = grid.cv_results_["mean_test_score"]
        alphas = grid.cv_results_["param_alpha"].tolist()

        # Plot results
        plt.suptitle("Lasso Regression: Optimization")
        fig, axs = plt.subplots(2,1,figsize=(10,10));
        axs[0].plot(alphas, mean_score)
        axs[0].set_xscale('log')
        axs[0].set_xlabel("Alpha")
        axs[0].set_ylabel("RMSE")

        # Get optimal value of alpha from first run
        alpha_1 = grid.best_estimator_.get_params()["alpha"]

        ## Second optimization
        model = Lasso()
        alpha_range_2 = np.linspace(alpha_1/10, alpha_1*10, num=iters_second) \
            .tolist() 

        param_grid = dict(alpha=alpha_range_2)
        
        # Use grid-search with cross-validation
        grid = GridSearchCV(
            model,
            param_grid,
            cv=self.my_cv,
            scoring='neg_root_mean_squared_error',
            n_jobs = self.n_jobs
        ) 
        grid.fit(self.train_inputs, self.train_targets)

        # Get results from grid search
        mean_score = grid.cv_results_["mean_test_score"]
        alphas = grid.cv_results_["param_alpha"].tolist() 
        alpha_2 = grid.best_estimator_.get_params()["alpha"]
        
        # Print results
        if(verbose == 1 or verbose == 2):
            print(f"Result from first run: alpha = {alpha_1}")
            print(f"Result from second run: alpha = {alpha_2}")

        # Plot results
        axs[1].plot(alphas, mean_score)
        axs[1].set_xscale('linear')
        axs[1].set_xlabel("Alpha")
        axs[1].set_ylabel("RMSE")
        
        fig.tight_layout()
        
        # Save plot
        if save_fig:
            plt.savefig(
                save_location + save_name +"_tuning" + ".svg",
                format="svg",
                bbox_inches="tight"
            )
            
        # Save grid results as .txt
        if save_grid_results:
            ModelEvaluator.grid_to_txt(
                grid,
                save_location=save_location,
                save_name=save_name + "_grid_results.txt"
            )

        # Plot the model
        if(verbose == 2):
            plt.show()
        else:
            plt.close(fig)

        # Optimized model
        optimized_model = grid.best_estimator_

        # Get parity plots with the optimized model
        results,metrics = self.train_and_evaluate(
            optimized_model,
            fig_title="Lasso Regression",
            save_fig=save_fig,
            save_location=save_location,
            save_name=save_name,
            verbose=verbose)
        
        return grid, (results, metrics)


    ## Ridge Regression

    def tune_and_plot_ridge(
        self,
        start_alphas,
        iters_second=1000,
        save_fig=False,
        save_grid_results=False,
        save_location="",
        save_name="Ridge",
        verbose=2):
        """ Optimizes the hyperparameter alpha from ridge regression.
            The optimal value of alpha is determined in two steps. Plots 
            curves with cross-validation RMSE of the optimizations and plots
            the parity plots of predictions from the trained models. 

        Parameters:
        ---------

        start_alphas (List<int>): Range with alphas to start the search with.

        save_fig (Boolean): Whether to save the figure. Defaults to False.
        
        save_grid_results (Boolen): Whether to save results of hyperparameter
            tuning as text file. Defaults to False.

        iters_second (int): Number of points for second optimization.
            Defaults to 1000.

        save_location (String): Directory to save the figures. Defaults to
            current directory

        save_name (String): Name of the saved figure. Defaults to "Figure".
        
        verbose (int): Verbosity of the output: 
            2: Print results and show parity plot.
            1: Show parity plot.
            0: Do not print and do not show parity plot. 
            Defaults to 2.

        Returns: 
        ---------

        tuple(grid (sklearn.model_selection.GridSearchCV),
            tuple(results, metrics)):
            grid: Results of hyperparameter optimization. The optimized models
            and metrics for the optimization can be obtained from this grid.
             
            (results, metrics): See return of self.train_and_evaluate().

        """
        model = Ridge()

        ## First optimization
        print("Optimizing Ridge regression model:")
        param_grid = dict(alpha=start_alphas)
        grid = GridSearchCV(
            model,
            param_grid,
            cv=self.my_cv,
            scoring='neg_root_mean_squared_error',
            n_jobs = self.n_jobs) # Use grid-search with cross-validation
        grid.fit(self.train_inputs, self.train_targets)

        # Get results from grid search
        mean_score = grid.cv_results_["mean_test_score"]
        alphas = grid.cv_results_["param_alpha"].tolist()

        # Plot results
        plt.suptitle("Ridge Regression: Optimization")
        fig, axs = plt.subplots(2,1,figsize=(10,10));
        axs[0].plot(alphas, mean_score)
        axs[0].set_xscale('log')
        axs[0].set_xlabel("Alpha")
        axs[0].set_ylabel("RMSE")


        # Get optimal value of alpha from first run
        alpha_1 = grid.best_estimator_.get_params()["alpha"]

        ## Second optimization
        model = Ridge()

        alpha_range_2 = np.linspace(alpha_1/10, alpha_1*10, num=iters_second) \
            .tolist()

        # Use grid-search with cross-validation
        param_grid = dict(alpha=alpha_range_2)
        grid = GridSearchCV(
            model,
            param_grid,
            cv=self.my_cv,
            scoring='neg_root_mean_squared_error',
            n_jobs = self.n_jobs
        )        
        
        grid.fit(self.train_inputs, self.train_targets)

        # Get results from grid search
        mean_score = grid.cv_results_["mean_test_score"]
        alphas = grid.cv_results_["param_alpha"].tolist()
        alpha_2 = grid.best_estimator_.get_params()["alpha"]
        
        # Print results
        if(verbose == 1 or verbose == 2):
            print(f"Result from first run: alpha = {alpha_1}")
            print(f"Result from second run: alpha = {alpha_2}")

        # Plot results
        axs[1].plot(alphas, mean_score)
        axs[1].set_xscale('linear')
        axs[1].set_xlabel("Alpha")
        axs[1].set_ylabel("RMSE")
        
        fig.tight_layout()
        
        # Save plot
        if save_fig:
            plt.savefig(
                save_location + save_name +"_tuning" + ".svg",
                format="svg",
                bbox_inches="tight"
            )

        # Plot the model
        if(verbose == 2):
            plt.show()
        else:
            plt.close(fig)
            
        # Save grid results as .txt
        if save_grid_results:
            ModelEvaluator.grid_to_txt(
                grid,
                save_location=save_location,
                save_name=save_name + "_grid_results.txt"
            )
            
        # Optimized model
        optimized_model = grid.best_estimator_

        # Get parity plots with the optimized model
        results,metrics = self.train_and_evaluate(
            optimized_model,
            fig_title="Ridge Regression",
            save_fig=save_fig,
            save_location=save_location,
            save_name=save_name,
            verbose=verbose)
        
        return grid, (results, metrics)


    ## RandomForest
    # Randomized search
    def random_searchCV_rf(
        self,
        random_grid,
        rs=None,
        n_iters=70
        ):
        """ Function to perform randomized search with Random Forest
            regression model to optimize hyperparameters.

        Parameters:        
        ---------
        
        random_grid (dict): Name of hyperparameters and their ranges 
            to look for optimization.
            
        rs (int): Random state for the search and the RF models. Defaults to
            None, so the internal rs of the modeltrainer is used.

        n_iters (int): Number of steps to search for optimization. Defaults
            to 70.

        Returns:
        ---------

        rf_random (sklearn.model_selection.RandomizedSearchCV): Grid after
        hyperparameter optimization. The optimized models and metrics for the
        optimization can be obtained from this grid. 

        """  
        if(rs is None):
            rs = self.rs
        
        model = RandomForestRegressor(
            random_state=rs,
            n_jobs=self.n_jobs,
        )

        # Randomized search
        rf_random = RandomizedSearchCV(
            estimator=model,
            param_distributions=random_grid,
            n_iter=n_iters,
            cv=self.my_cv,
            verbose=0,
            random_state=rs,
            scoring='neg_root_mean_squared_error',
            n_jobs=self.n_jobs,
            )

        rf_random.fit(self.train_inputs, self.train_targets)

        return rf_random

         
    def tune_and_plot_rf_random(
        self,
        random_grid,
        n_iters=70,
        rs=None,
        save_fig=False,
        save_grid_results=False,
        save_location="",
        save_name="Random_Forest_RandomSearchCV",
        verbose=2
        ):
        """ Tune Random Forest regression model with randomized search and plot
            the results of the optimized models in a parity plot.

        Parameters:
        ---------

        random_grid (dict): Name of hyperparameters and their ranges 
          for optimization.
          
        rs (int): Random state for the search and the RF models. Defaults to
            None, so the internal rs of the modeltrainer is used.

        save_fig (boolean): Whether to save the figure. Defaults to False.
        
        save_grid_results (Boolen): Whether to save results of hyperparameter
            tuning as text file. Defaults to False.

        save_location (String): Directory to save the figure. Defaults to
            current directory.

        save_name (String): Name of the saved figure. Defaults to
            "Random_Forest_RandomSearchCV".
            
        verbose (int): Verbosity of the output: 
            2: Print results and show parity plot.
            1: Show parity plot.
            0: Do not print and do not show parity plot. 
            Defaults to 2.
            

        Returns:
        ---------

        tuple(grid (sklearn.model_selection.RandomizedSearchCV),
            tuple(results, metrics)):
            grid: Results of hyperparameter optimization. The optimized models
            and metrics for the optimization can be obtained from this grid.
             
            (results, metrics): See return of self.train_and_evaluate().

        """
        if (rs is None):
            rs = self.rs
        
        # Get best Random Forest model from randomized search
        grid = self.random_searchCV_rf(
            random_grid,
            n_iters=n_iters,
            rs=rs
        )
        
        # Save grid results as .txt
        if save_grid_results:
            ModelEvaluator.grid_to_txt(
                grid,
                save_location=save_location,
                save_name=save_name + "_grid_results.txt"
            )

        optimized_model = grid.best_estimator_

        # Parity plots with the optimized model
        results,metrics = self.train_and_evaluate(
            optimized_model,
            fig_title="Random Forest Regression (Random_SearchCV)",
            save_fig=save_fig,
            save_location=save_location,
            save_name=save_name,
            verbose=verbose)

        return grid, (results, metrics)



    def grid_searchCV_rf(
        self,
        grid,
        rs=None
    ):
        """ Function to perform grid search with Random Forest
            regression model to optimize hyperparameters.

        Parameters:        
        ---------
        
        grid (dict): Name of hyperparameters and their ranges 
            for optimization.

        rs (int): Random state for the RF models. Defaults to None, 
            so the internal rs of the modeltrainer is used.
        
        Returns:
        ---------

        rf_grid (sklearn.model_selection.GridSearchCV): Grid after
            hyperparameter optimization. The optimized models and metrics for
            the optimization can be obtained from this grid. 

        """
        if (rs is None):
            rs = self.rs
        
        # Define model
        model = RandomForestRegressor(
            random_state=rs,
            n_jobs=self.n_jobs
            )

        # Grid search
        rf_grid = GridSearchCV(
            model,
            grid,
            cv=self.my_cv,
            verbose=0,
            scoring='neg_root_mean_squared_error',
            n_jobs=self.n_jobs)

        rf_grid.fit(self.train_inputs, self.train_targets)

        return rf_grid
        

    def tune_and_plot_rf_grid(
        self,
        random_grid,
        rs=None,
        save_fig=False,
        save_grid_results=False,
        save_location="",
        save_name="Random_Forest_GridSearchCV",
        verbose=2
        ):
        """ Tune Random Forest regression model with grid search and plot
            the results of the optimized models in a parity plot.

        Parameters:
        ---------

        random_grid (dict): Name of hyperparameters and their ranges 
          for optimization.
          
        rs (int): Random state for the search and the RF models. Defaults to
            None, so the internal rs of the modeltrainer is used.

        save_fig (boolean): Whether to save the figure. Defaults to False.
        
        save_grid_results (Boolen): Whether to save results of hyperparameter
            tuning as text file. Defaults to False.

        save_location (String): Directory to save the figure. Defaults to
            current directory.

        save_name (String): Name of the saved figure. Defaults to
            "Random_Forest_GridSearchCV".
            
        verbose (int): Verbosity of the output: 
            2: Print results and show parity plot.
            1: Show parity plot.
            0: Do not print and do not show parity plot. 
            Defaults to 2.

        Returns:
        ---------

        tuple(grid (sklearn.model_selection.GridSearchCV),
            tuple(results, metrics)):
            grid: Results of hyperparameter optimization. The optimized models
            and metrics for the optimization can be obtained from this grid.
             
            (results, metrics): See return of self.train_and_evaluate().

        """
        if (rs is None):
            rs = self.rs
        
        # Get best Random Forest model from Random Search
        grid = self.grid_searchCV_rf(random_grid, rs=rs)
        optimized_model = grid.best_estimator_

        # Save grid results as .txt
        if save_grid_results:
            ModelEvaluator.grid_to_txt(
                grid,
                save_location=save_location,
                save_name=save_name + "_grid_results.txt"
            )

        # Get parity plots with the optimized model
        results,metrics = self.train_and_evaluate(
            optimized_model,
            fig_title="Random Forest Regression (Grid_SearchCV)",
            save_fig=save_fig,
            save_location=save_location,
            save_name=save_name,
            verbose=verbose
        )
        
        # Return optimized model
        return grid, (results, metrics)


    ## Keras sequential
    def get_model_layers(self):
        """ Returns a list of a list of layers for different Keras Sequential
        models from file in same directory.
        """
        return get_layer_list(self.train_inputs.shape[1])


    def build_sequential(
        self,
        layer_list,
        optimizer="rmsprop",
        loss="mse",
        metrics=["mae"]
        ):
        """ Creates and returns a Keras Sequential model from given List of 
            Keras layers.

        Parameters:        
        ---------
        
        layer_list (List<Keras-Layer>): Keras layers for the model in order 
            how it will be created in the model.

        optimizer (String): Optimizer used to train the model. 
            Defaults to "rmsprop".

        loss (String): Metric for loss function. Defaults to "mse".

        metrics (List<String>): Metrics used for evaluation. 

        Returns:
        ---------

        model (Keras.Sequential): Compiled Keras model.

        """
        # Sequential model for feed-forward NN
        model = keras.models.Sequential()

        # Add layers to the model
        for layer in layer_list:
            model.add(layer)

        # Compile and return model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)
        
        return model


    def tune_sequential_epochs(
        self,
        layers_list=None,
        cv=4,
        max_epochs=300
        ):
        """ Train different Keras sequential models from given layers
            and plot the cross-validation accuracy (MAE) vs. training epochs.

        Parameters:        
        ---------
        
        layers_list (List<Keras-Layer>): List of list of Keras layers for each 
            model. Defaults to None. Thereby, the inbuilt method
            get_model_layers is used.

        cv (int): cross-validation for the optimization. Defaults to 4. 

        max_epochs (int): Number of epochs to train the models. Defaults to 300.


        Returns:
        ---------

        MAE_histories (List<List<float>>): MAE vs. Epoch for each model. 

        """
        # Build models from given layers
        if layers_list is None:
            layers_list = self.get_model_layers()

        model_list = [
            self.build_sequential(model_layers) for model_layers in layers_list
            ]

        # Use k fold cross validation to optimize the models
        k = cv
        num_val_samples = len(self.train_inputs) // k

        # Inner function to optain the optimal number of epochs
        def get_epoch_error(model, num_epochs=max_epochs):
            # List for mae for each epoch
            all_mae_histories = []

            for i in range(k):
                print(f"Run #{i}")

                # Get cross-validation data from training data
                cv_data = self.train_inputs[
                    i*num_val_samples : (i+1)*num_val_samples
                    ]
                cv_targets = self.train_targets[
                    i*num_val_samples : (i+1)*num_val_samples
                    ]

                # Get training data in k folds
                partial_train_data = np.concatenate([
                    self.train_inputs[:i * num_val_samples],
                    self.train_inputs[(i+1) * num_val_samples:]
                ], axis=0)
                
                partial_train_targets = np.concatenate([
                    self.train_targets[:i * num_val_samples],
                    self.train_targets[(i+1) * num_val_samples:]
                ], axis=0)

                # Fit model using cross-validation
                history = model.fit(
                    partial_train_data,
                    partial_train_targets,
                    validation_data=(cv_data, cv_targets),
                    epochs=num_epochs,
                    batch_size=1,
                    verbose=0
                )

                # Get MAE from history
                mae_history = history.history["val_mae"]
                all_mae_histories.append(mae_history)

            # Get average of MAE for each epoch and for each fold
            average_mae_history = [
                np.mean([x[i] for x in all_mae_histories])
                for i in range(num_epochs)
            ]

            return average_mae_history

        # Get mean-absolute-errors for each model and each epoch
        MAE_histories = []

        for i, model in enumerate(model_list):
            print(f"Model {i}")
            MAE_histories.append(get_epoch_error(model))


        # Plot MAE vs. epochs for all models in separate plots for each model
        print(f"""Results from training of the models
        with {k} fold cross-validation:""")
        for i, history in enumerate(MAE_histories):
            self.plot_sequential_MAE_history(
                history,
                max_epochs=max_epochs,
                fig_title=f"Model {i}")

        # Return MAE vs. epoch count
        return MAE_histories

    
    def plot_sequential_MAE_history(
        self,
        history,
        fig_title=""):
        """ Plot the results from Keras sequential optimization
            of the optimal number of epochs (from function 
            tune_sequential_model_epochs).

        Parameters:        
        ---------
        
        history (List<float>): MAE after each epoch.

        fig_title (String): Title displayed in the plot. Defaults to "".

        """
        # Create new plots everytime this function is called
        plt.figure()
        plt.plot(range(1, len(history) +1), history)

        # Adjust style of the plot
        plt.xticks(np.arange(0,len(history),50))
        plt.minorticks_on()
        plt.title(fig_title)


    # Get smoother curves
    def plot_sequential_MAE_history_smooth(
        self,
        MAE_histories,
        factor=0.95):
        """ Method similar plot_sequential_MAE_history, but with 
            smoothened curves.

        """

        # Innter Function to smooth given points to smoothen noisy line.
        def smooth_curve(
            points,
            factor=factor):
            smoothed_points = []

            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(
                        previous * factor + point * (1-factor)
                        )
                
                else:
                    smoothed_points.append(point)

            return smoothed_points


        # Get smoothened points from MAE history
        smoothed_MAE_histories = [
            smooth_curve(history[10:]) for history in MAE_histories
            ]

        # Plot smoothened line from MAE history
        for i, history in enumerate(smoothed_MAE_histories):
            self.plot_sequential_MAE_history(
                history,
                fig_title=f"Model {i}")

        # Return smoothened points from MAE histories
        return smoothed_MAE_histories


    def train_Sequential_with_epochs(
        self,
        epoch_list,
        layers_list=None
        ):
        """ Trains the different Keras Sequential models from given layer list
            and given number of epochs. Should be used after determining the
            optimal number of epochs.

        Parameters:
        ---------

        epoch_list (List<int>): Number of epochs for each model.

        layers_list (List<Keras-Layer>): List of list of Keras layers for each 
        model. Defaults to None. Thereby, the inbuilt method get_model_layers
        is used.

        Returns:
        ---------

        model_list (List<Keras.Sequential>): Trained models.

        """

        # Build models from given layers
        if layers_list is None:
            layers_list = self.get_model_layers()

        model_list = [
            self.build_sequential(model_layers) for model_layers in layers_list
            ]

        # Train the models
        for model, num_epoch in zip(model_list, epoch_list):
            model.fit(
                self.train_inputs,
                self.train_targets,
                epochs=num_epoch,
                batch_size=1,
                verbose= 0)
                
        # Print metrics and plot parity plots
        for i, model in enumerate(model_list):
            print(f"Model {i}:")
            self.evaluate(
                model,
                fig_title=f"Multilayer Perceptron {i}"
            )

        # Return trained models
        return model_list
    
    
    ## Reference models
    def train_and_evaluate_AveragePredictor(
        self,
        **kwargs
    ):
        """ Trains and evaluates AveragePredictor model as a reference model.
        
            For kwargs and returns, see method train_and_evaluate.
        """
        
        model = AveragePredictor()
        results,metrics = self.train_and_evaluate(model, **kwargs)
        
        return results, metrics
        
        


class Modeltrainer_train_test(ModelTrainer_base):
    """ Inherits from Modeltrainer_base. Uses separate training and test sets
        for training, optimization, and evaluation of the models.

        Attributes
        ----------
        
        my_cv (int or sklearn.model_selection.LeaveOneOut): (class attribute)
            Cross-validation folds. Defaults to Leave-One-Out (scikit-learn).

        n_jobs (int): (class attribute) Number of parallel jobs running
            (-1 to run all possible jobs). WARNING: Sometimes setting n_jobs to
            -1 could cause errors. 

        train_lim (tuple<float,float>): (class attribute) Lower and upper limit
            for plotting the training set. If it is set to None, the limits are
            determined automatically every time. Defaults to None.

        test_lim (tuple<float,float>): (class attribute) Lower and upper limit
            for plotting the test set.  If it is set to None, the limits are
            determined automatically every time. Defaults to None.

        train_inputs (pandas.DataFrame): Features of training set.

        train_targets (pandas.Series): Labels of training set.
        
        test_inputs (pandas.DataFrame): Features of test set. Defaults to None
        
        rs (int): Random-state for methods that use
            randomness. Defaults to 0.
            
        test_targets (pandas.Series): Labels of test set. Defaults to None

        is_log (boolean): Set to true, if the label of dataset has been 
            transformed to log(10). Defaults to False.
            
        label_name (String): Name of the label. Automatically taken from
        train_targets.
        
        label_display (String): Name to display the labels in the parity plots.
            Defaults to label_name.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    ## Methods
    def train_and_evaluate(
        self,
        model,
        fig_title="",
        font_scale=1.5,
        save_fig=False,
        save_location="",
        save_name="Figure",
        verbose=2
    ):
        """ Train a given model with the dataset stored as attributes.
            Evaluate the trained model by printing accuracy metrics of the 
            predictions for training set and for test set (MAE, RMSE, and R2).
            Furthermore, plot a parity plot with the predictions from the
            training set and the test set.
        
        Parameters:
        ----------

        model (from scikit-learn or Keras): Regression model to train.

        fig_title (String): Title of the parity plot. Defaults to "".

        font_scale (float): Font scale in the plots from seaborn.set_context. 
            Defaults to 1.5.

        save_fig (Boolean): Whether to save the figure. Defaults to False.

        save_location (String): Directory to save the figure. Defaults to
            current directory.

        save_name (String): Filename to save. Defaults to "Figure".
        
        verbose (int): Verbosity of the output: 
            2: Print results and show parity plot.
            1: Show parity plot.
            0: Do not print and do not show parity plot. 
            Defaults to 2.

        Returns: 
        ---------
        (results_dict, metrics_dict) (tuple(dict, dict):
            results_dict: Contains train_predictions (pd.Series),
                test_predictions (pd.Series), and trained and optimized model.
                If the label is_log is true: there are 
                train_predictions_original and test_predictions_original as 
                additional entries.
            
            metrics_dict: Contains metrics of the model: RMSE training, 
                RMSE test, MAE training, MAE test, R2 score training, R2 score
                test.

        """
        # Train model
        model.fit(self.train_inputs, self.train_targets)
        
        # Create predictions
        train_preds = model.predict(self.train_inputs)
        test_preds = model.predict(self.test_inputs)

        # Flatten results if given model is a Keras model
        if isinstance(model, keras.engine.sequential.Sequential):
            train_preds = train_preds.flatten()
            test_preds = test_preds.flatten()

        # Calculate RMSE, MAE, and r2 score
        train_rmse = ModelEvaluator.RMSE(train_preds, self.train_targets)
        train_mae = ModelEvaluator.MAE(train_preds, self.train_targets)
        train_r2 = r2_score(self.train_targets, train_preds)

        test_rmse = ModelEvaluator.RMSE(test_preds, self.test_targets)
        test_mae = ModelEvaluator.MAE(test_preds, self.test_targets)
        test_r2 = r2_score(self.test_targets, test_preds)
        
        # Whether to display plot
        show_plot = False
        if(verbose == 2):
            show_plot = True
            
        # Get parity plot of the predictions
        PlotTools.parity_plot_train_test(
            train_preds,
            test_preds,
            self.train_targets,
            self.test_targets,
            fig_title=fig_title,
            train_lim=self.train_lim,
            test_lim=self.test_lim,
            font_scale=font_scale,
            is_log=self.is_log,
            save_fig=save_fig,
            label_display=self.label_display,
            save_location=save_location,
            save_name=save_name,
            show_plot=show_plot)
        
        if (verbose == 1 or verbose == 2):
            # Print accuracy of the predictions
            print(f"""RMSE on training set: {train_rmse},
            MAE on training set: {train_mae}
            r2-score on training set: {train_r2}""")
            
            print(f"""RMSE on validation set: {test_rmse},
            MAE on validation set: {test_mae}
            r2-score on validation set: {test_r2}""")

        # If label has been transformed to log before to training, report
        # errors without scaling (original values)
        if self.is_log:
            train_preds_normal = np.power(10, train_preds)
            test_preds_normal = np.power(10, test_preds)
            train_targets_normal = np.power(10, self.train_targets)
            test_targets_normal = np.power(10, self.test_targets)

            train_rmse_normal = ModelEvaluator.RMSE(
                train_preds_normal,
                train_targets_normal)

            train_mae_normal = ModelEvaluator.MAE(
                train_preds_normal,
                train_targets_normal)

            train_r2_normal = r2_score(
                train_targets_normal,
                train_preds_normal)

            test_rmse_normal = ModelEvaluator.RMSE(
                test_preds_normal,
                test_targets_normal)

            test_mae_normal = ModelEvaluator.MAE(
                test_preds_normal,
                test_targets_normal)

            test_r2_normal = r2_score(
                test_targets_normal,
                test_preds_normal)

            if(verbose == 1 or verbose == 2):
                print("Adjusting log10(y) -> y:")
                # Print accuracy of the predictions
                print(f"""RMSE on training set: {train_rmse_normal},
                MAE on training set: {train_mae_normal}
                r2-score on training set: {train_r2_normal}""")
                
                print(f"""RMSE on validation set: {test_rmse_normal},
                MAE on validation set: {test_mae_normal}
                r2-score on validation set: {test_r2_normal}""")
            
        # Add results to dictionaries for return
        results_dict = {
            "train_predictions": train_preds,
            "test-predictions": test_preds,
            "model": model
        }
        
        metrics_dict = {
            "RMSE_training": train_rmse,
            "MAE training": train_mae,
            "R2_training": train_r2,
            "RMSE_test": test_rmse,
            "MAE_test": test_mae,
            "R2_test": test_r2
        }
        
         # Add original values (log(y) -> y) to results dictionary
        if self.is_log:
            results_dict["train_predictions_original"] = train_preds_normal
            results_dict["test_predictions_original"] = test_preds_normal
            
            metrics_dict["RMSE_training_original"] = train_rmse_normal
            metrics_dict["MAE_training_original"] = train_mae_normal
            metrics_dict["R2_training_original"] = train_r2_normal
            metrics_dict["RMSE_test_original"] = test_rmse_normal
            metrics_dict["MAE_test_original"] = test_mae_normal
            metrics_dict["R2_test_original"] = test_r2_normal

        # Return predictions
        return results_dict, metrics_dict


    def evaluate(
        self,
        model,
        fig_title="",
        font_scale=1.5,
        save_fig=False,
        save_location="",
        save_name="Figure",
        verbose=2):
        """ Evaluate an already trained model by printing accuracy metrics of
            the predictions for training set and for test set (MAE, RMSE,
            and R2). Furthermore, plot a parity plot with the predictions from
            the training set and the test set.

        Parameters:
        ----------

        model (from scikit-learn or keras): Regression model to train.

        fig_title (String): Title of the parity plot. Defaults to "".

        font_scale (float): Font scale in the plots from seaborn.set_context. 
            Defaults to 1.5.

        save_fig (Boolean): Whether to save the figure. Defaults to False.

        save_location (String): Directory to save the figure. Defaults to
            current directory.

        save_name (String): Filename to save. Defaults to "Figure".
        
        verbose (int): Verbosity of the output: 
            2: Print results and show parity plot.
            1: Show parity plot.
            0: Do not print and do not show parity plot. 
            Defaults to 2.

        Returns: 
        ---------
        (results_dict, metrics_dict) (tuple(dict, dict):
            results_dict: Contains train_predictions (pd.Series),
                test_predictions (pd.Series), and trained and optimized model.
                If the label is_log is true: there are 
                train_predictions_original and test_predictions_original as 
                additional entries.
            
            metrics_dict: Contains metrics of the model: RMSE training, 
                RMSE test, MAE training, MAE test, R2 score training, R2 score
                test.

        """
        # Create predictions
        train_preds = model.predict(self.train_inputs)
        test_preds = model.predict(self.test_inputs)

        # Flatten results if given model is a Keras model
        if isinstance(model, keras.engine.sequential.Sequential):
            train_preds = train_preds.flatten()
            test_preds = test_preds.flatten()

        # Calculate RMSE, MAE, and r2 score
        train_rmse = ModelEvaluator.RMSE(train_preds, self.train_targets)
        train_mae = ModelEvaluator.MAE(train_preds, self.train_targets)
        train_r2 = r2_score(self.train_targets, train_preds)

        test_rmse = ModelEvaluator.RMSE(test_preds, self.test_targets)
        test_mae = ModelEvaluator.MAE(test_preds, self.test_targets)
        test_r2 = r2_score(self.test_targets, test_preds)

        # Whether to show plot
        show_plot = False
        
        if(verbose == 2):
            show_plot = True
        # Get parity plot of the predictions
        PlotTools.parity_plot_train_test(
            train_preds,
            test_preds,
            self.train_targets,
            self.test_targets,
            fig_title=fig_title,
            train_lim=self.train_lim,
            test_lim=self.test_lim,
            font_scale=font_scale,
            is_log=self.is_log,
            label_display=self.label_display,
            save_fig=save_fig,
            save_location=save_location,
            save_name=save_name,
            show_plot=show_plot)
            
        
        if(verbose == 1 or verbose == 2):
            # Print accuracy of the predictions
            print(f"""RMSE on training set: {train_rmse},
            MAE on training set: {train_mae}
            r2-score on training set: {train_r2}""")
            
            print(f"""RMSE on validation set: {test_rmse},
            MAE on validation set: {test_mae}
            r2-score on validation set: {test_r2}""")

        # If label has been transformed to log prior to training, report
        # Errors without scaling (original values):
        if self.is_log:
            train_preds_normal = np.power(10, train_preds)
            test_preds_normal = np.power(10, test_preds)
            train_targets_normal = np.power(10, self.train_targets)
            test_targets_normal = np.power(10, self.test_targets)

            train_rmse_normal = ModelEvaluator.RMSE(
                train_preds_normal,
                train_targets_normal)

            train_mae_normal = ModelEvaluator.MAE(
                train_preds_normal,
                train_targets_normal)

            train_r2_normal = r2_score(
                train_targets_normal,
                train_preds_normal)

            test_rmse_normal = ModelEvaluator.RMSE(
                test_preds_normal,
                test_targets_normal)

            test_mae_normal = ModelEvaluator.MAE(
                test_preds_normal,
                test_targets_normal)

            test_r2_normal = r2_score(
                test_targets_normal,
                test_preds_normal)
            
            if(verbose == 1 or verbose == 2):
                print("Adjusting log(y) -> y:")
                # Print accuracy of the predictions
                print(f"""RMSE on training set: {train_rmse_normal},
                MAE on training set: {train_mae_normal}
                r2-score on training set: {train_r2_normal}""")
                
                print(f"""RMSE on validation set: {test_rmse_normal},
                MAE on validation set: {test_mae_normal}
                r2-score on validation set: {test_r2_normal}""")

        # Add results to dictionaries for return
        results_dict = {
            "train_predictions": train_preds,
            "test-predictions": test_preds,
            "model": model
        }
        
        metrics_dict = {
            "RMSE_training": train_rmse,
            "MAE training": train_mae,
            "R2_training": train_r2,
            "RMSE_test": test_rmse,
            "MAE_test": test_mae,
            "R2_test": test_r2
        }
        
         # Add original values (log(y) -> y) to results dictionary
        if self.is_log:
            results_dict["train_predictions_original"] = train_preds_normal
            results_dict["test_predictions_original"] = test_preds_normal
            
            metrics_dict["RMSE_training_original"] = train_rmse_normal
            metrics_dict["MAE_training_original"] = train_mae_normal
            metrics_dict["R2_training_original"] = train_r2_normal
            metrics_dict["RMSE_test_original"] = test_rmse_normal
            metrics_dict["MAE_test_original"] = test_mae_normal
            metrics_dict["R2_test_original"] = test_r2_normal

        # Return predictions
        return results_dict, metrics_dict


class Modeltrainer_LOO(ModelTrainer_base):
    """ Inherits from Modeltrainer_base. Uses only training set, and LOO-CV for
        evaluation of the models

        TO-DO: Continute implementation.
    """
    def __init__(
        self,
        train_inputs,
        train_targets,
        is_log=False,
        rs=0):

        super().__init__(
            train_inputs,
            train_targets,
            test_inputs=None,
            test_targets=None,
            is_log=is_log,
            rs=rs
        )
      


