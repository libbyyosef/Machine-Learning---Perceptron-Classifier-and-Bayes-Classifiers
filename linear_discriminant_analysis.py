from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in
        `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in
        `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, \
            None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector,
        same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, self.pi_ = np.unique(y, return_counts=True)
        self.pi_ = self.pi_ / len(y)
        self.mu_ = []
        for single_class in self.classes_:
            X_class = X[y == single_class]
            mean_class = np.mean(X_class, axis=0)
            self.mu_.append(mean_class)
        self.mu_ = np.array(self.mu_)
        y_converts_to_ints = y.astype(int)
        normalize_factor=len(X) - len(self.classes_)
        x_minus_mean = X - self.mu_[y_converts_to_ints]
        self.cov_ = np.dot(x_minus_mean.T, x_minus_mean) / normalize_factor
        self.cov_inv_ = np.linalg.inv(self.cov_)

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
        n_features = X.shape[1]
        double_pi = 2 * np.pi
        det_cov = np.linalg.det(self.cov_)
        z = np.sqrt(np.power(double_pi, n_features) * det_cov)
        d = X[:, np.newaxis] - self.mu_
        l = np.exp(-0.5 * np.sum(d @ self.cov_inv_ * d, axis=2))
        l *= (self.pi_ / z)
        return l

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
