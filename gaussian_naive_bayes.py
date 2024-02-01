from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in
            `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, self.pi_ = np.unique(y, return_counts=True)
        self.pi_=self.pi_/len(y)
        self.mu_, self.vars_ = [], []
        for single_class in self.classes_:
            X_class = X[y==single_class ]
            mean_class = np.mean(X_class, axis=0)
            self.mu_.append(mean_class)
        self.mu_ = np.array(self.mu_)
        for single_class in self.classes_:
            X_class = X[y == single_class]
            var_class = np.var(X_class, axis=0, ddof=1)
            self.vars_.append(var_class)
        self.vars_ = np.array(self.vars_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihoods = self.likelihood(X)
        max_vals_indexes = np.argmax(likelihoods, axis=1)
        return self.classes_[max_vals_indexes]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` "
                "function")
        x_minus_mean=X[:, np.newaxis, :] - self.mu_
        mone=np.exp(np.power(x_minus_mean,2) / (- 2 * self.vars_))
        double_pi=2*np.pi
        l = mone / np.sqrt(double_pi * self.vars_)
        likelihoods = np.exp(np.sum(np.log(l), axis=2)) * self.pi_
        return likelihoods


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_predict = self.predict(X)
        return misclassification_error(y_true=y, y_pred=y_predict)

