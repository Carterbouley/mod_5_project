# Credit Card Default Prediction 

This project is a part of the Flatiron Bootcamp London Immersive course. 

#### -- Project Status: Completed

## Project Intro/Objective

The purpose of this project is predict credit card defaults based on previous financial activity. This reduces the workload on financial institutions, and helps them appropriately manage their credit card risk.

### Partner
* Zaria Rankine
* Website for partner
* Partner contact: [Name of Contact], [slack handle of contact if any]
* If you do not have a partner leave this section out

### Methods Used
* Machine Learning
* Data Visualization
* Predictive Modeling
* Feature Engineering

### Technologies
* Python.
* Pandas, jupyter
* Logistic Regression
* K Nearest Neighbours
* Random Forest
* Bagged Decision Tree
* PCA
* Feature Engineering

## Project Description

The notebook to be analysed is called 'Final Notebook'

The dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
Our aim was to use solely this dataset to predict weather or not someone would default on their credit card debt.

Firstly we performed some data exploration, breaking down the dataset by age, education, gender and marital status. Then we engineered some extra features, finding average percentage of bill paid, how much of the assigned credit limit people were using (following the logic that someone with $9,000 out of a $10,000 credit limit is closer to default than $9,000 out of $1,000,000 credit limit), average payments etc. We also added polynomials, to get a feel for feature interactions, such as young unmarried uneducated male vs old married educated female.

A simple decision tree was used as our baseline model, leading to an AUC score of 0.6152

The dataset was quite imbalanced, so we fixed that through upsampling. We perfomed a SVM before and after this to see the difference in results.

Our logisitc regression resulted in an ROC AUC of 0.7070, with a threshold of 0.5

The SVM gridsearch is highly computationally intensive, and so with poor hyperparameters it didn't perform particularly well

Simiilarly our K Nearest Neighbours wasn't very good either, with an AUC score of 0.6137

Random Forest score was 0.7572, leading us to beleive that trees were the most effective ways of classifying this data. We explored further options, leading us to the Bagged Tree. 

Bagged Tree score was 0.8475, our final and more effective model.

## Needs of this project

- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
- Presentation


