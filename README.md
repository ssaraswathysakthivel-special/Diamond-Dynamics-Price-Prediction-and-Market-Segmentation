Project Overview
An interactive web app that predicts diamond prices and identifies the market segment of a diamond using machine learning models.
Regression (ANN): Predicts diamond price in INR.
Clustering (KMeans): Identifies diamond category for business insights.
🎯 Ideal for diamond sellers, buyers, and analytics dashboards.

Features
1️⃣ Price Prediction Module
Predict diamond price based on:
Numeric: carat, x, y, z
Categorical: cut, color, clarity
Button: Predict Price
Output: Price in INR

2️⃣ Market Segment Prediction Module
Predicts market segment:
Premium Heavy Diamonds
Mid-range Standard Diamonds
Affordable Small Diamonds
Same inputs as price prediction
Output: Cluster ID + human-readable name

3️⃣ Interactive UI
3-column layout for input form
Styled gradient header for premium look
Clear output boxes for predictions

Tech Stack
Python 3.10+
Streamlit for web interface
TensorFlow (Keras) for ANN regression

Dependencies (requirements.txt)
streamlit
tensorflow
numpy
pandas
scikit-learn
joblibscikit-learn for KMeans clustering

NumPy & Pandas for data processing
