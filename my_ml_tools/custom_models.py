import numpy as np


class AveragePredictor:
    """Reference model that always returns the average value of the labels in
    the training set as predictions.

    Attributes
    ----------

    train_targets_mean (numpy.float64): Mean value of the labels in the
        training set.

    """

    def __init__(self):
        self.train_targets_mean = None

    def fit(self, train_inputs, train_targets):
        """Fit the model by calculating the average value of the labels in the
        training set.

        Parameters
        ----------
        train_inputs (pandas.DataFrame): DataFrame containing the features
            of the training set. This is not needed for this model, but
            including this parameter facilitates the interoperability of
            this methods with the other methods from the Modeltrainer class.

        train_targets(pandas.Series): Series containing the labels of the
            training set.

        """
        # Get mean value of training labels. This value represents
        # Each prediction
        self.train_targets_mean = train_targets.mean()

    def predict(self, inputs):
        """Create predictions from the inputs.

        Parameters
        ----------
            inputs (pandas.DataFrame): DataFrame containing the features
                of the training set.

        Returns
        ----------
            preds (numpy.ndarray): Predictions of the model.

        """
        # Get length of inputs
        len_inputs = inputs.shape[0]

        # Create lists that represent the predictions
        if self.train_targets_mean is None:
            raise (
                Exception(
                    """ERROR! Model has to be fit before creating
                                predictions!"""
                )
            )

        preds_list = len_inputs * [self.train_targets_mean]

        # Return result as numpy array
        preds = np.asarray(preds_list)
        return preds
