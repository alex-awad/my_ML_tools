""" Package with methods to facilitate plotting """

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
matplotlib.use('Agg') 
import seaborn as sns
import numpy as np

class PlotTools:

    def __init__(self):
        ...

    
    @staticmethod
    def boxplot_and_hist(
        df,
        column,
        binwidth=None,
        binrange=None,
        hist_minor_ticks=None,
        alt_xlabel=None,
        save_fig=False,
        save_location="",
        fig_name="XX",
        filetype="svg",
        font_scale=1.5,
        height_ratio=(.10, .90),
        figsize=(10,7.5)
        ):
        """ Creates a plot including a histogram and a boxplot from a specific
            column in a Pandas DataFrame.

        Parameters:
        ----------

        df (pandas.DataFrame): Dataframe containing the data.

        column (String): Name of the column that contains the values to
            be plotted.

        binwidth (float): Width of the bins in the histogram. Defaults to None 
            (automatically determined).

        binrange (tuple<float,float>): Minimum and maximum limit of histogram.
            Defaults to None. Thus, it is automatically identified.
            
        hist_minor_ticks (float): Interval of minor ticks to be added in histogram.
            Defaults to None, so no minor ticks are set.

        alt_xlabel (String): Alternative label for the column to be displayed 
            in the plots. Defaults to None. Thus, the label represents the 
            column name. 

        save_fig (Boolean): Whether to save the figure. Defaults to False.

        save_location (String): Directory where the figure is saved. Defaults to
          the current directory.

        fig_name (String): Filename of the figure for saving. Defaults to "XX".

        filetype (String): Filetype to save the figure. Defaults to "svg".

        font_scale (float): Scale of the font for seaborn.set_context. Defaults
            to 1.5.

        height_ratio (Tuple<float,float>): Ratio of the size of boxplot and
            histogram in the figure. Defaults to (0.10, 0.90).

        figsize (Tuple<float, float>): Size of the figure. Defaults to (10,7.5).

        """
        # Set plot options
        sns.set_context("notebook", font_scale=font_scale)

        # Create figure
        f, (ax_box, ax_hist) = plt.subplots(
            2, sharex=True,
            gridspec_kw={"height_ratios": height_ratio},
            figsize=figsize,
            dpi=300)

        # Add boxplot and histogram
        sns.boxplot(
            data=df,
            x=column,
            ax=ax_box)

        sns.histplot(
            data=df,
            x=column,
            ax=ax_hist,
            binwidth=binwidth,
            binrange=binrange,
            edgecolor="black",
            linewidth=0.5)
        
        if hist_minor_ticks is not None:
            ax_hist.xaxis.set_minor_locator(MultipleLocator(hist_minor_ticks))

        # Change aperance of the boxplot
        for i,box in enumerate(ax_box.artists):
            box.set_edgecolor('black')
            box.set_facecolor('white')
        ax_box.set(yticks=[], xlabel="")

        # Change title of x-axis
        if (alt_xlabel is not None):
            ax_hist.set_xlabel(alt_xlabel)

        # Save figure
        if save_fig:
            f.savefig(
                f"{save_location}/{fig_name}.{filetype}",
                format=filetype,
                bbox_inches='tight')

        # Reset plot options
        sns.set_context("notebook", font_scale=1.0)



    @staticmethod
    def get_limits(predictions, targets):
        """ Automatically returns limits of the predictions and targets
            for the parity plot. Works both for train limits and test limits.

            Lower limit is the lowest point from all points with an extra
            margin. Upper limit is the highest point from all points.

        Parameters:
        ----------
            predictions (pandas.Series): Predictions from the models.

            targets (pandas.Series): Actual labels. 

        Returns:
        ----------

        (Tuple<float, float>): Lower limit and upper limit for parity plots.

        """
        # Get lowest point
        min_value = predictions.min() if predictions.min() < targets.min() \
            else targets.min()

        # Get highest point
        max_testue = predictions.max() if predictions.max() > targets.max() \
            else targets.max()

        # Get higher and lower absolute value
        higher_value = abs(min_value) if abs(min_value) > abs(max_testue) \
            else abs(max_testue)

        lower_value = abs(min_value) if abs(min_value) < abs(max_testue) \
            else abs(max_testue)

        # Get difference between higher and lower absolute value
        diff = higher_value - lower_value

        # Return limits with an extra margin for better plots
        return(
            min_value - 0.1*abs(diff),
            max_testue + 0.1*abs(diff)
            )



    @staticmethod
    def parity_plot_train_test(
    tr_predictions,
    tes_predictions,
    tr_targets,
    tes_targets,
    train_lim=None,
    test_lim=None,
    label_display=None,
    fig_title="",
    is_log=False,
    font_scale=1.85,
    save_fig=False,
    save_location="",
    save_name="Figure",
    show_plot=True
    ):
        """ Creates parity plot for predicted labels vs. actual labels for
            both the training set and the test set.

        Parameters:
        ------------

        tr_predictions (pandas.Series): Predictions of training set.

        tes_predictions (pandas.Series): Predictions of test set.

        tr_targets (pandas.Series): Labels of training set.

        tes_targets (pandas.Series) Labels of test set.

        train_lim (Tuple<float,float>): Min and max limit of the parity plot for
            training set. Defaults to None. Thereby, limits are determined 
            automatically dependent on the given data.

        test_lim (Tuple<float,float>): Min and max limit of the parity plot for
            test set. Defaults to None. Thereby, limits are determined 
            automatically dependent on the given data.
            
        label_display (String): Displayed name of label in plots.
            Defaults to None, so label name is automatically determined from 
            name of train_targets.

        fig_title (String): Title of the figure. Defaults to "".

        is_log (Boolean): Whether the labels are logarithmically scaled.
            Defaults to False.

        font_scale (float): Font scale for seaborn.set_context. Defaults
            to 1.85.

        save_fig (Boolean): Whether to save the figure. Defaults to False.

        save_location (String): Directory to save the figure. Defaults to
            current directory.

        save_name (String): Name of the saved figure. Defaults to "Figure".
        
        show_plot (Boolean): Whether to display the plot. Defaults to True.

        """
        sns.set_context("notebook", font_scale=font_scale)

        # For straight line y = m*x
        x = np.linspace(-10,10)
        y = x

        # Automatically determine train and test limits if they are not set
        if(train_lim is None):
            train_lim = PlotTools.get_limits(tr_predictions, tr_targets)
        
        if (test_lim is None):
            test_lim = PlotTools.get_limits(tes_predictions, tes_targets)

        # Create plots
        f, (ax_train, ax_test) = plt.subplots(1,2, figsize=(15, 7.5), dpi=80)
        sns.scatterplot(x=tr_targets, y=tr_predictions, ax=ax_train, s=70)
        sns.scatterplot(
            x=tes_targets,
            y=tes_predictions,
            ax=ax_test,
            color="g",
            s=70)
        ax_train.plot(x,y, c="r")
        ax_test.plot(x,y, c="r")

        # Set figure title
        f.suptitle(fig_title, fontsize=28)

        # Set subplot titles
        ax_train.title.set_text("Training set")
        ax_test.title.set_text("Test set")

        if( (train_lim is not None) and (test_lim is not None)):
            # Check if plot limits are outside of given range
            if(tr_predictions.min() < train_lim[0] or \
                tr_targets.min() < train_lim[0]):

                print("Train limit or predictions are below plot limits!")
            
            if(tr_predictions.max() > train_lim[1] or \
                tr_targets.max() > train_lim[1]):

                print("Train limit or predictions are above plot limits!")

            if(tes_predictions.min() < test_lim[0] or \
                tes_targets.min() < test_lim[0]):

                print("Test limit or predictions are below plot limits!")
            
            if(tes_predictions.max() > test_lim[1] or \
                tes_targets.max() > test_lim[1]):

                print("Test limit or predictions are above plot limits!")

        # Set name of label in plots
        if (label_display is None):
            label_name = tr_targets.name
        else:
            label_name = label_display
        
        # Axis settings
        ax_train.axis('square')  
        ax_train.set_xlim(*train_lim)
        ax_train.set_ylim(*train_lim)
        ax_train.set_xlabel(f"True {label_name}")
        ax_train.set_ylabel(f"Predicted {label_name}")

        ax_test.axis("square")  
        ax_test.set_xlim(*test_lim)
        ax_test.set_ylim(*test_lim)
        ax_test.set_xlabel(f"True {label_name}")
        ax_test.set_ylabel(f"Predicted {label_name}")

        f.tight_layout()

        # Save figure
        if(save_fig):
            plt.savefig(
                save_location + save_name + ".svg",
                format="svg",
                bbox_inches="tight"
            )

        
        # Reset font size
        sns.set_context("notebook", font_scale=1.0)
        
        # Whether to show plot
        if(show_plot):
            plt.show()
        else:
            plt.close(f)
        
        
    @staticmethod
    def parity_plot_single(
        predictions,
        targets,
        ax_lim=None,
        label_display=None,
        fig_title="",
        is_log=False,
        font_scale=1.85,
        save_fig=False,
        save_location="",
        save_name="Figure"
    ):
        """ Creates parity plot for predicted labels vs. actual labels for
            only one set.

        Parameters:
        ------------

        predictions(pandas.Series): Predictions.

        targets (pandas.Series): Labels.

        ax_lim (Tuple<float,float>): Min and max limit of the parity plot for
            the set. Defaults to None. Thereby, limits are determined 
            automatically dependent on the given data.

        fig_title (String): Title of the figure. Defaults to "".
        
        label_display (String): Displayed name of label in plots.
            Defaults to None, so label name is automatically determined from 
            name of train_targets.

        is_log (Boolean): Whether the labels are logarithmically scaled.
            Defaults to False.

        font_scale (float): Font scale for seaborn.set_context. Defaults
            to 1.85.

        save_fig (Boolean): Whether to save the figure. Defaults to False.

        save_location (String): Directory to save the figure. Defaults to
            current directory.

        save_name (String): Name of the saved figure. Defaults to "Figure".

        """
        sns.set_context("notebook", font_scale=font_scale)

        # For straight line y = m*x
        x = np.linspace(-10,10)
        y = x
        
        # Automatically determine axis limits if they are not set
        if(ax_lim is None):
            ax_lim = PlotTools.get_limits(predictions, targets)

        # Create plots
        fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
        sns.scatterplot(
            x=targets,
            y=predictions,
            ax=ax,
            s=70
        )
        ax.plot(x, y, c="r")

        # Set figure title
        fig.suptitle(fig_title, fontsize=28)

        if(ax_lim is not None):
            # Check if plot limits are outside of given range
            if(predictions.min() < ax_lim[0]):
                print("Limit of dataset is below plot limits!")
            
            if(predictions.max() > ax_lim[1]):
                print("Limit of dataset is above plot limits!")

        # Set name of label in plots
        if (label_display is None):
            label_name = targets.name
        else:
            label_name = label_display

        # Axis settings
        ax.axis('square')  
        ax.set_xlim(*ax_lim)
        ax.set_ylim(*ax_lim)
        ax.set_xlabel(f"True {label_name}")
        ax.set_ylabel(f"True {label_name}")
            
        fig.tight_layout()

        # Save figure
        if(save_fig):
            plt.savefig(
                save_location + save_name + ".svg",
                format="svg",
                bbox_inches="tight"
            )

        # Reset font size
        sns.set_context("notebook", font_scale=1.0)
        
        


