## House Price Prediction Project
Hey there! In this project, I've taken on the exciting task of predicting house prices using machine learning techniques. I completed this project as part of my internship at Bharat Intern.

### Introduction
I started off by loading a dataset from an Excel file named "HousePricePrediction.xlsx." The goal here is to create a model that can predict house prices accurately using the given data.

### Libraries I Used
For this project, I relied on a few powerful tools:

Pandas: I used this library to handle and analyze the dataset efficiently.
Seaborn: This handy tool helped me visualize the data using different graphs and plots.
Scikit-learn: I turned to this library for various tasks like preprocessing the data, training the model, and evaluating its performance.
Matplotlib: With this library, I could create informative visualizations that helped me understand the dataset better.
Random Forest Regressor: This is the algorithm I chose for the main part of the project – predicting house prices.

### My Journey
Here's how I tackled the project step by step:

### Data Exploration
I began by loading the dataset and taking a peek at the first five records using Pandas. To understand the data better, I calculated its shape (the number of rows and columns) and identified the different types of variables.

### Data Cleaning
Next, I had to clean up the dataset. I dropped the 'Id' column as it wasn't going to be useful for prediction. I also took care of missing values in the 'SalePrice' column by filling them with the mean value. To make things even smoother, I removed any remaining rows with missing data.

### Data Preprocessing
Before I could feed the data into a machine learning model, I needed to convert categorical variables into a numerical format. I achieved this using a technique called One-Hot Encoding.

### Model Training
Time to dive into the exciting part – training the model! I split the dataset into a training set and a validation set using Scikit-learn's train_test_split function. The model I chose to use is called Random Forest Regressor, which is great for regression tasks like predicting house prices. I trained the model using the training data.

### Model Evaluation
With the model trained, I put it to the test. I predicted house prices on the validation set and calculated the Mean Absolute Percentage Error (MAPE) to see how well the model was performing.

### Conclusion
In a nutshell, my project involved predicting house prices using a Random Forest Regressor model. I started by exploring and cleaning the dataset, then preprocessing the data for the model. Finally, I trained the model and evaluated its performance using the MAPE metric.

Feel free to check out the detailed code and comments in the Jupyter Notebook file for more insights into my journey during this internship at Bharat Intern.
