import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


class DataLoader:
    """ Class to load in data from either a pandas dataframe or an excel file.
    Includes several methods to pre-process the given data, such as splitting
    the data into training and test sets, scaling and data cleanup.
    
    Attributes
    ----------
    
    df (pandas.DataFrame): DataFrame containing the read in data.
    
    rs (int): Random-state for methods using random-seeds. Defaults to 0.

    scaler (sklearn.preprocessing.*): Scaler from package sklearn.preprocessing
        to scale the input features.

    feature_df (pandas.DataFrame): DataFrame containing the columns representing
        the featrues.

    label_series (pandas.Series): Series containing the label.

    label_name (String): Name of the label.

    train_inputs (pandas.DataFrame): Dataframe containing the split features 
        from training set.
    
    test_inputs: 
    
    train_targets,
    
    test_targets
    """
    
    def __init__(
        self,
        df,
        rs=0
    ):
        # Check if pandas.DataFrame is passed as constructor
        if isinstance(df, pd.core.frame.DataFrame):
            self.df = df
        else:
            raise TypeError("""Argument has to be a pandas dataframe! \n
            To parse from files, use the inbuilt class methods!""")

        self.rs = rs
        
        # Add placeholders for attributes that are assigned later
        self.scaler = None
        self.feature_df = None
        self.label_series = None
        self.label_name = None
        self.train_inputs = None
        self.test_inputs = None
        self.train_targets = None
        self.test_targets = None


    ## Alternative constructors
    @classmethod
    def from_excel(
        cls,
        filepath,
        **kwargs
    ):
        """Creates instance from filepath to an excel file."""
        return cls(pd.read_excel(filepath), **kwargs)

    
    # Static methods
    @staticmethod 
    def load_model(path):
        """Loads serialized models and returns them. 
        Works wih models from scikit-learn that were serialized using pickle.

        Parameters:
        ----------
        path (String): Path to the serialized model.

        Returns:
        ----------
        model: Serialized model from sklearn. 

        """
        # Load serialized model
        with open(path, 'rb') as pickle_model:
            model = pickle.load(pickle_model)
            pickle_model.close()

        return model


    ## Get methods
    def get_df(self):
        return self.df


    ## Utility methods
    def are_na_entries(self):
        """Checks if there are entries with missing or nan-values.

        Returns:
        ---------
        (Boolean): True if there are nan-values.
            False if there are no missing values.
        """

        # Series with number of nan entries per column in the dataframe
        na_series = self.df.isna().sum()
        return(na_series[na_series.values > 0].size > 0)


    ## Data cleanup methods
    def remove_equal_cols(self):
        """Remove columns where all fields have the same values.
        
        Returns:
        ---------
        self.df: DataFrame attribute.
        """
        n_unique = self.df.nunique()
        drop_cols = n_unique[n_unique == 1].index
        self.df = self.df.drop(drop_cols, axis=1)
        return self.df


    ## Split dataframe methods
    def split_features_label(
        self,
        label_name,
        ignore_features=None):
        """Split features and label from internal dataframe to two 
        separate dataframe and series.

        Parameters:
        ----------
        label_name (String): Name of the label denoted in the internal
            dataframe.

        ignore_features (List<String>): List of features that are excluded in
            the returned dataframe containing the features. Defaults to None.

        Returns:
        ----------
        self.feature_df, self.label_series: Attributes of instance.
        
        """
        # Get columns from df as list
        cols = self.df.columns.to_list()

        # Get feature columns
        feature_cols = cols.copy()
        feature_cols.remove(label_name) # Remove label
        
        # Remove additional features from keyword-argument
        if ignore_features is not None:
            if isinstance(ignore_features, list):
                for feature in ignore_features:
                    feature_cols.remove(feature)

            else:
                feature_cols.remove(ignore_features)    
        
        # Get label column
        self.label_name = label_name
        label_col = label_name

        # Get dataframes and series 
        self.feature_df = self.df[feature_cols].copy()
        self.label_series = self.df[label_col].copy()

        # Return feature df and label series
        return self.feature_df, self.label_series


    def combine_feature_label(self):
            """Combine separate feature df and label series to a single 
            dataframe. Overwrites the current dataframe!

            This method only works after having applied the method 
            "split_features_label" before.
            
            Returns:
            ---------
            self.df: Attributes of instance.
            """
            if self.feature_df is None or self.label_series is None:
                raise ValueError(
                    "The features and labels have not been split yet")
            else:
                self.df = self.feature_df.assign(
                    **{self.label_name: self.label_series})

            # Return dataframe
            return self.df

    def split_train_test(
        self,
        label_name,
        ignore_features=None,
        test_size=0.25):
        """Uses inbuilt functions to facilitate the train_test_split
        method from scikit-learn.
        
        Parameters:
        ----------
        label_name (String): Name of the label denoted in the internal
            dataframe.

        ignore_features (List[String]): List of features that are excluded in
            the returned dataframe containing the features. Defaults to None.

        test_size (float): Fraction of samples to put into test set.

        Returns:
        ----------
        (tuple): (self.train_inputs, self.test_inputs, self.train_targets, 
            self.test_targets): Dataframes contaning split training and test
            sets with features and labels separated."""

        # Split features and labels
        self.split_features_label(label_name)

        # Remove features from inputs
        if ignore_features is not None:
            remove_feature_df = self.feature_df.drop(
                ignore_features, axis=1
            )
        else:
            remove_feature_df = self.feature_df

        # Split inputs and targets (features and labels for ML)
        (self.train_inputs,
        self.test_inputs,
        self.train_targets,
        self.test_targets) = train_test_split(
            remove_feature_df,
            self.label_series,
            test_size=test_size,
            random_state=self.rs)
        
        # Return split dataset
        return (self.train_inputs,
        self.test_inputs,
        self.train_targets,
        self.test_targets) 


    def scale_inputs(self, scaler=None):
        """Scale the inputs with a given scaler. 
        It is important to fit the scaler only to the training set,
        not with the validation or test set!

        Parameters:
        ----------
        scaler: New scaler object from sklearn, if no scaler is passed
            use the internal scaler.
        """
        # Check if scaler is passed as keyword argument
        if scaler is not None:
            self.scaler = scaler
            self.scaler = scaler.fit(self.train_inputs[:])

        else:
            print("No new scaler passed, using loaded scaler!")

        # Transform train inputs and test inputs
        self.train_inputs[:] = self.scaler.transform(self.train_inputs[:])
        self.test_inputs[:] = self.scaler.transform(self.test_inputs[:])


    def pickle_scaler(self, path):
        """Serialize scaler with pickle.

        Parameters:
        ----------
        path (String): Path for output of pickled scaler.
        """
        with open(path, 'wb') as file:
            pickle.dump(self.scaler, file)
            file.close()


    def scaler_from_pickle(self, path):
        """Load serialized scaler.

        Parameters:
        ----------
        path (String): Path of pickled scaler.
        """
        with open(path, 'rb') as pickle_model:
            self.scaler = pickle.load(pickle_model)

