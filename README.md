# FUTURE_ML_02

# Sentiment Analysis on Movie Reviews

This project involves building a **Natural Language Processing (NLP)** pipeline to perform sentiment analysis on a dataset of movie reviews. The primary objective is to classify reviews as either positive or negative. Various steps, from data preprocessing to model evaluation, are detailed in this project.

## Features

- **Data Exploration**:  
  Includes visualizations of sentiment distribution and common word frequencies in both positive and negative reviews using tools like Matplotlib, Seaborn, and Plotly.

- **Data Preprocessing**:  
  - Removal of HTML tags, special characters, and stopwords.  
  - Text normalization through stemming and tokenization.  
  - Deduplication of reviews for cleaner data.  

- **Feature Engineering**:  
  TF-IDF vectorization to convert text data into numerical form for model training.

- **Model Building**:  
  Implemented and compared multiple machine learning models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - XGBoost

- **Model Evaluation**:  
  Evaluated models using metrics like accuracy, confusion matrices, and classification reports. Visualization of confusion matrices is included for better interpretation.

- **Hyperparameter Tuning**:  
  Used GridSearchCV for optimizing model performance and ensuring robustness.

## Tools and Libraries

- **Programming Language**: Python  
- **Libraries**:  
  - NLP: NLTK, SpaCy, WordCloud, Contractions  
  - Machine Learning: Scikit-learn, XGBoost  
  - Data Visualization: Matplotlib, Seaborn, Plotly  
  - Data Manipulation: Pandas, NumPy

## Results

The project demonstrates the effectiveness of various machine learning models for sentiment analysis.  
- The **Support Vector Machine (SVM)** model achieved the highest accuracy of **90%**.  
- The models were evaluated with detailed classification reports and confusion matrices.

## How to Use

1. Clone this repository:  
   ```bash
   git clone https://github.com/vinnienovah/sentiment-analysis-movie-reviews.git
   cd sentiment-analysis-movie-reviews
   
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Run the Jupyter Notebook or script to preprocess data, train models, and evaluate performance.

## Dataset
The datasets used in this project are provided by the [Kaggle: IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).


## Author
This project was created as part of a machine learning learning initiative. Contributions, suggestions, or feedback are welcome!
