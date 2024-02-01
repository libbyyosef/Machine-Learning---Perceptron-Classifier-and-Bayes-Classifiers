# Perceptron-Classifier-and-Bayes-Classifiers

**Perceptron Classifier and Bayes Classifiers**

This project focuses on implementing both the Perceptron classifier and Bayes classifiers.

**Implemented Functionalities:**

**Perceptron Classifier:**

- **Misclassification Error Function:** Implemented the misclassification_error function in the **metrics/loss_functions.py** file. This function calculates the misclassification error based on the predictions made by the Perceptron classifier.
- **Perceptron Algorithm:** Implemented the Perceptron algorithm in the **learners/classifiers/perceptron.py** file. The Perceptron algorithm is a linear classifier that iteratively learns the optimal weights to separate classes in the training data. The misclassification error function implemented above is utilized in the toy implementation to evaluate the performance of the Perceptron.

**Bayes Classifiers:**

- **Accuracy Function**: Implemented the accuracy function in the **metrics/loss_functions.py** file. This function calculates the accuracy of predictions made by the Bayes classifiers.
- **LDA Classifier:** Implemented the Linear Discriminant Analysis (LDA) classifier in the - -**learners/classifiers/linear_discriminant_analysis.py** file. LDA is a classification algorithm that models the distribution of each class as a Gaussian distribution and uses linear decision boundaries to separate classes.
- **Gaussian Naive Bayes Classifier:** Implemented the Gaussian Naive Bayes classifier in the **learners/classifiers/gaussian_naive_bayes.py** file. This classifier assumes that features are independent and follows a Gaussian distribution, making it suitable for classification tasks with continuous features.

**Summary**

This project provides implementations of both the Perceptron classifier and Bayes classifiers, enabling the classification of data based on these algorithms. The implemented functionalities include misclassification error calculation, Perceptron algorithm implementation, accuracy calculation, and implementation of LDA and Gaussian Naive Bayes classifiers. These implementations offer a comprehensive approach to building and evaluating machine learning models for classification tasks.
