SMS Spam Detection Project 📧
This project focuses on building and evaluating a machine learning model to classify SMS messages as either spam or ham (not spam). The project includes a Jupyter Notebook detailing the entire development process and a Streamlit web application for real-time classification.

📈 Project Workflow
The project follows a systematic pipeline, fully documented in sms-spam-detection.ipynb:
Data Cleaning:
Loaded the dataset from spam.csv.
Removed extraneous columns and renamed the essential columns to target and text.
Converted the categorical target variable ('ham'/'spam') into numerical format (0/1).
Identified and removed duplicate entries to ensure data quality.
Exploratory Data Analysis (EDA):
Analyzed the class distribution, revealing an imbalance between ham and spam messages.
Engineered new features: num_characters, num_words, and num_sentences.
Performed a statistical comparison of these new features across both classes, noting that spam messages tend to be longer.
Visualized data distributions and word frequencies using histograms and word clouds.
Text Preprocessing:
A custom function was created to clean and normalize the text data. This pipeline includes:
Lowercasing
Tokenization
Alphanumeric Filtering
Stopword Removal
Stemming
Model Building & Evaluation:
Vectorization: Used the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text into numerical feature vectors.
Model Comparison: Trained and evaluated several classification algorithms, with Multinomial Naive Bayes (MNB) being selected for its high accuracy and perfect precision.

Model Export: Saved the trained TF-IDF vectorizer and the MNB model as vectorizer.pkl and sms_spam_model.pkl for use in the web app.

🚀 Technologies Used
Python
Streamlit: For building the interactive web application.
Scikit-learn: For model training and evaluation.
NLTK (Natural Language Toolkit): For text preprocessing.
Pandas & NumPy: For data manipulation.
Matplotlib & Seaborn: For data visualization.
Jupyter Notebook: As the development environment.