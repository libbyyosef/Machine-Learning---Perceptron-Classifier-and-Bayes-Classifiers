import pandas as pd

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import matplotlib.pyplot as plt
from IMLearn.learners.classifiers import perceptron


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent
    features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss
    values (y-axis)
    as a function of the training iterations (x-axis).
    """
    # separable - hard, inseparable - soft
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=lambda per, _, __: losses.append(
            per.loss(X, y)))
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        df = pd.DataFrame({'Iteration': range(len(losses)), 'Loss': losses})
        layout = go.Layout(
            title={"text": f"Perceptron Training Error - {n} dataset",
                   "x": 0.5},
            xaxis={"title": "Iteration"},
            yaxis={"title": "Loss - Misclassification Error"})
        figure = go.Figure(data=go.Scatter(x=df['Iteration'], y=df['Loss'],
                                           mode="lines",
                                           marker=dict(color="black")),
                           layout=layout)
        figure.write_image(f"perceptron_fit_iter_{n}.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified
    covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and
    gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        lda_predicts = lda.predict(X)
        g_naive_bayes = GaussianNaiveBayes().fit(X, y)
        g_naive_bayes_predicts = g_naive_bayes.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = accuracy(y, lda_predicts) * 100
        g_naive_accuracy = accuracy(y, g_naive_bayes_predicts) * 100
        round_lda_accuracy = round(lda_accuracy, )
        round_g_naive_accuracy = round(g_naive_accuracy, 2)
        graph = make_subplots(rows=1, cols=2,
                              subplot_titles=(
                                  rf"$\text{{Gaussian Naive Bayes (accuracy="
                                  rf"{round_g_naive_accuracy}%)}}$",
                                  rf"$\text{{LDA - linear discriminant \
        analysis(accuracy={round_lda_accuracy}%)}}$"))

        # Add traces for data-points setting symbols and colors
        scatter_lda = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                 marker=dict(color=lda_predicts,
                                             symbol=class_symbols[y],
                                             colorscale=class_colors(3)))
        scatter_g_naive_bayes = go.Scatter(x=X[:, 0], y=X[:, 1],
                                           mode='markers',
                                           marker=dict(
                                               color=g_naive_bayes_predicts,
                                               symbol=class_symbols[y],
                                               colorscale=class_colors(3)))
        graph.add_trace(scatter_g_naive_bayes, row=1, col=1)
        graph.add_trace(scatter_lda, row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        scatter_lda_mean = go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                      mode="markers",
                                      marker=dict(symbol="x", color="black",
                                                  size=15))
        scatter_g_naive_bayes_mean = go.Scatter(x=g_naive_bayes.mu_[:, 0],
                                                y=g_naive_bayes.mu_[:, 1],
                                                mode="markers",
                                                marker=dict(symbol="x",
                                                            color="black",
                                                            size=15))
        graph.add_trace(scatter_g_naive_bayes_mean, row=1, col=1)
        graph.add_trace(scatter_lda_mean, row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            ellipse_g_naive_bayes = get_ellipse(g_naive_bayes.mu_[i], np.diag(
                g_naive_bayes.vars_[i]))
            ellipse_lda = get_ellipse(lda.mu_[i], lda.cov_)
            graph.add_trace(ellipse_g_naive_bayes, row=1, col=1)
            graph.add_trace(ellipse_lda, row=1, col=2)
        graph.update_yaxes(scaleratio=1, scaleanchor="x")
        graph.update_layout(
            title_text=rf"$\text{{Comparing Gaussian Classifiers - {f[:-4]}dataset}}$",
            width=800, height=400, showlegend=False)
        file_name=f"lda_vs_gaussian_naive.bayes.{f[:-4]}.png"
        graph.write_image(file_name)


if __name__ == '__main__':
    np.random.seed(0)
run_perceptron()
compare_gaussian_classifiers()
