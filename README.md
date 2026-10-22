# Python Machine Learning Assignment

*Submitted by:* [Lorna awino okoth]  
*Project Title:* Customer Segmentation and Predictive Analysis  

---

## 📋 Project Descriptions

### 1. Supervised Learning Project (supervised_ml.py)
This project utilizes a Supervised Machine Learning model to make predictions based on labeled historical training data. It implements a *Linear Regression* algorithm to understand patterns and forecast target numeric outputs based on given input features.

### 2. Unsupervised Learning Project (unsupervised_ml.py)
This project implements an Unsupervised Machine Learning model to discover hidden groupings within unlabeled data. It uses the *K-Means Clustering* algorithm to automatically segment 10 supermarket customers into distinct behavioral groups based on their *Monthly Spending (KES)* and *Monthly Visits. It also includes a visual data scatter plot generated using **Matplotlib*.

---

## 🛠️ Installation & Setup Instructions

### Prerequisites
Make sure you have *Python 3* installed on your system.

### 1. Install Dependencies
Open your terminal inside the project directory and run the following command to install the required libraries (scikit-learn and matplotlib):
bash
python -m pip install sklearn matplotlib


### 2. How to Run the Code

To run the Unsupervised Learning clustering model and view the scatter plot, execute:
bash
python unsupervised_ml.py


To run the Supervised Learning model script, execute:
bash
python supervised_ml.py


---

## 🧠 Part C: Theory Questions & Answers

### 1. What is supervised machine learning?
Supervised machine learning is a type of machine learning where the model is trained on a *labeled dataset*. This means that for every input sample in the training data, the correct, corresponding output (target label) is already provided. 

### 2. What is unsupervised machine learning?
Unsupervised machine learning is a type of machine learning where the model is given *unlabeled data* without any pre-assigned target outputs. 

### 3. Explain the difference between supervised and unsupervised learning.
* *Supervised Learning* uses labeled data to train a model to predict specific targets or classifications. Its main goal is to map data to a known target variable.
* *Unsupervised Learning* uses unlabeled data to explore data structures and find natural relationships or densities. Its main goal is to understand data distribution or group data items based on inherent similarities.

### 4. Why is Linear Regression considered supervised learning?
Linear Regression is considered supervised learning because it requires explicit *training labels (historical continuous values)* to learn the mathematical relationships between independent features (\(X\)) and the dependent target variable (\(y\)). It calculates an error metric based on how close its prediction is to the actual, known ground-truth value.

### 5. Why is K-Means considered unsupervised learning?
K-Means is considered unsupervised learning because it groups data points purely by measuring spatial distances between features, completely *without any target categories or correct answers*. It optimizes cluster centroids automatically based solely on the distribution of data points.

### 6. What is the difference between fit() and predict()?
* *fit(): This method executes the **training phase*. It analyzes the input data to learn parameters (such as the mean/variance for scaling, coefficients for regression, or cluster centers for K-Means).
* *predict(): This method executes the *inference phae. Once the model parameters are learned via 'fit()', 'predict()' uses that knowledge to assign output values or cluster IDs to new or existing data points.

### 7. Give two real-world applications of supervised learning and two of unsupervised learning.

#### Supervised Learning Applications:
1. *House Price Prediction:* Forecasting a property's market value based on features like square footage, neighborhood, and bedroom count.
2. *Email Spam Detection:* Classifying incoming emails as either "Spam" or "Not Spam" based on historical word patterns.

#### Unsupervised Learning Applications:
1. *Market Customer Segmentation:* Grouping supermarket buyers by spending behavior and visit frequency to design targeted marketing campaigns.
2. *Anomaly Detection:* Flagging unusual credit card transaction patterns to identify potential fraudulent behavior.
