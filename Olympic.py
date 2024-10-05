import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


a = pd.read_csv('athletes new.csv')

st.markdown("""
    <style>
        .main {
            font-family: 'Sans-serif';
        }
        h1 {
            color: #4a4a4a;
        }
        .stSidebar {
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("Data Insights & Machine Learning")
    
    # Data Description
    st.subheader("Data Overview")
    data_selection = st.selectbox("Select Data Insight", ["Head", "Tail", "Shape", "Info", "Describe"])

    # Feature Engineering
    st.subheader("Feature Engineering")
    feature_selection = st.selectbox("Select Feature Insight", ["Feature Correlation", "Distribution", "Count Plots"])

    # ML Algorithm
    st.subheader("Machine Learning Algorithms")
    algorithm = st.selectbox("Choose Algorithm", 
                             ["Logistic Regression", "Random Forest", "Gradient Descent", "Decision Tree"])

# Display  main page
st.title("Athlete Dataset Analysis & Predictions")

# Data Overview 
if data_selection == "Head":
    st.write("## Dataset Head")
    st.dataframe(a.head())
elif data_selection == "Tail":
    st.write("## Dataset Tail")
    st.dataframe(a.tail())
elif data_selection == "Shape":
    st.write("## Dataset Shape")
    st.write(a.shape)
elif data_selection == "Info":
    st.write("## Dataset Info")
    st.text(a.info())
elif data_selection == "Describe":
    st.write("## Dataset Description")
    st.write(a.describe())

# Feature Insights
n = ['height', 'weight']

if feature_selection == "Feature Correlation":
    st.write("## Correlation Heatmap")
    plt.figure(figsize=(10,8))
    sns.heatmap(a[n].corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
elif feature_selection == "Distribution":
    for feature in n:
        st.write(f"## {feature} Distribution")
        plt.figure(figsize=(10,5))
        sns.histplot(a[feature], kde=True)
        st.pyplot(plt)
elif feature_selection == "Count Plots":
    f = ['gender', 'disciplines', 'events', 'country']
    for feature in f:
        st.write(f"## {feature} Count Plot")
        plt.figure(figsize=(10,5))
        sns.countplot(data=a, x=feature, order=a[feature].value_counts().index)
        plt.xticks(rotation=90)
        st.pyplot(plt)

# Encoding & Data Preprocessing
le = LabelEncoder()
for col in ['gender', 'disciplines', 'events', 'country']:
    a[col] = le.fit_transform(a[col])

x = a[['gender', 'disciplines', 'events', 'country']]
y = a['weight_class'] = a['weight'].apply(lambda x: 'Underweight' if x < 50 else 'MiddleWeight' if x <= 70 else 'Overweight')
y = LabelEncoder().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model Selection and Training
if algorithm == "Logistic Regression":
    model = LogisticRegression()
elif algorithm == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif algorithm == "Gradient Descent":
    model = SGDClassifier()
elif algorithm == "Decision Tree":
    model = DecisionTreeClassifier()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Display Accuracy
st.markdown("## Prediction Results")
accuracy = accuracy_score(y_test, y_pred)
st.markdown(f"**Algorithm Used:**<span style='color: red; font-size: 35px;'> {algorithm}</span>", unsafe_allow_html=True)
st.markdown(f"**Accuracy:** <span style='color: green; font-size: 24px;'>{accuracy:.2f}</span>", unsafe_allow_html=True)

# Classification Report
st.write("## Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
st.write("## Confusion Matrix")
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
st.pyplot(plt)

# Feature Importance Plot for Random Forest, and Decision Tree
if algorithm in ["Random Forest", "Decision Tree"]:
    st.write("## Feature Importance")
    plt.figure(figsize=(10,8))
    sns.barplot(x=model.feature_importances_, y=x.columns)
    plt.title('Feature Importance')
    st.pyplot(plt)

# Set the layout to 80% of the screen size
st.markdown("""
    <style>
        .css-1d391kg {max-width: 80%; }
    </style>
    """, unsafe_allow_html=True)
