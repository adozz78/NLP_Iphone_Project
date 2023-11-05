# NLP Classification Project - iPhone SE Reviews and Ratings

**Author:** Adrien Ozanne  
**Email:** adrien.ozanne@epfedu.fr

## Project Overview

Welcome to the NLP Classification Project! This project focuses on analyzing and classifying reviews of the iPhone SE from an Indian e-commerce website. The main objectives of this project are as follows:

1. **Data Exploration:** Perform an exploratory data analysis on the dataset to understand its structure

2. **Baseline Models:** Implement baseline machine learning models to classify reviews into different ratings categories.

3. **Model Improvement:** Improve the baseline models by using various techniques

4. **Sequential Model (deep learning):** Implement and train a sequential model

5. **Comparison and Conclusion:** Compare the models' performances and draw conclusions 

## Dataset

The dataset used for this project contains reviews and ratings of the iPhone SE, available from [Kaggle](https://www.kaggle.com/datasets/kmldas/apple-iphone-se-reviews-ratings). The dataset includes user reviews and their corresponding ratings.

## Project Structure

The project is organized into the following notebooks and scripts:

1. **1_exploratory_data_analysis.ipynb:** An  exploration of the dataset, examining distribution, relationships and statistics.

2. **2_baseline_model.ipynb:** Implementation of baseline machine learning models for ratings classification.

3. **3_baseline_model_improve.ipynb:** Improvement of the baseline models through advanced techniques and hyperparameter tuning.

3. **4_deep_learning_model.ipynb:** Implementation of a simple RNN sequential model.

4. **preprocessing_pipeline.py:** A Python script for data preprocessing.
   

## Model Performances

The table below displays the performance metrics of different models implemented in this project:

| Model Name                  | Accuracy |
|-----------------------------|-----------|
| Gradient Boosting (Baseline)|       71%    |        
| Logistic Regression (Baseline)|    72%      |       
| Multinomial Naive Bayes (Baseline)|   70%  |        
| Gradient Boosting (Class redistribution)|    90%    |       
| Gradient Boosting (Class redistribution + Oversampling)|    81%    |  
| Gradient Boosting (Class redistribution + Undersampling)|    64%    |  
| Gradient Boosting (Class redistribution + Oversampling + tuning hyperparameters)|    85%    |        
| Sequential Model (RNN model)|       60%             |     
| Sequential Model (RNN model + Class redistribution + Oversampling)|       73%             |       

## Installation

To set up and run the project, follow these steps:

1. Clone the project repository:

   ```shell
   git clone https://github.com/adozz78/NLP_Iphone_Project
   cd NLP_Iphone_Project

2. Create a virtual environment (optional but recommended):

   ```shell
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'

3. Run the Jupyter notebooks or Python scripts 
   
## References

1. [Apple iPhone SE Reviews & Ratings (Kaggle)](https://www.kaggle.com/datasets/kmldas/apple-iphone-se-reviews-ratings)

2. [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

3. [Keras Documentation](https://keras.io/)




