import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import sys
from pandas.errors import ParserError
import time
import altair as altpi
import matplotlib.cm as cm
import graphviz
import base64
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Toggle, BoxAnnotation
from bokeh.models import Panel, Tabs
from bokeh.palettes import Set3


st.set_page_config(page_title='KwikML',layout="wide", page_icon="⚡")
hide_streamlit_style = """
            <style>
                        footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title('⚡ KwikML - A Machine Learning Workbench ⚡')
st.subheader("Helping you to get familiar with machine learning models directly from your web browser")

st.markdown(
        """
    - 🗂️ Upload dataset
    - ⚙️ Select a model and set its hyper-parameters
    - 📉 Train it and check its performance metrics and decision boundary on train and test data
    - 🩺 Diagnose possible ovewritting and experiment with other settings
    -----
    """
    )

# Main kwikml class
class kwikml:
    # Data preparation part, it will automatically handle your data
    def prepare_data(self, split_data, train_test):
        # Reduce data size
        data = self.data[self.features]
        data = data.sample(frac = round(split_data/100,2))

        # Impute nans with mean for numeris and most frequent for categoricals
        cat_imp = SimpleImputer(strategy="most_frequent")
        if len(data.loc[:,data.dtypes == 'object'].columns) != 0:
            data.loc[:,data.dtypes == 'object'] = cat_imp.fit_transform(data.loc[:,data.dtypes == 'object'])
        imp = SimpleImputer(missing_values = np.nan, strategy="mean")
        data.loc[:,data.dtypes != 'object'] = imp.fit_transform(data.loc[:,data.dtypes != 'object'])

        # One hot encoding for categorical variables
        cats = data.dtypes == 'object'
        le = LabelEncoder() 
        for x in data.columns[cats]:
            sum(pd.isna(data[x]))
            data.loc[:,x] = le.fit_transform(data[x])
        onehotencoder = OneHotEncoder() 
        data.loc[:, ~cats].join(pd.DataFrame(data=onehotencoder.fit_transform(data.loc[:,cats]).toarray(), columns= onehotencoder.get_feature_names()))

        # Set target column
        target_options = data.columns
        self.chosen_target = st.sidebar.selectbox("Select target column", (target_options))

        # Standardize the feature data
        X = data.loc[:, data.columns != self.chosen_target]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = data.loc[:, data.columns != self.chosen_target].columns
        y = data[self.chosen_target]

        # Train test split
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1 - train_test/100), random_state=42)
        except:
            st.markdown('<span style="color:red">With this amount of data and split size the train data will have no records, <br /> Please change reduce and split parameter <br /> </span>', unsafe_allow_html=True)  

    # Classifier type and algorithm selection 
    def set_classifier_properties(self):
        self.type = st.sidebar.selectbox("Algorithm type", ("Classification", "Regression"))
        if self.type == "Regression":
            self.chosen_classifier = st.sidebar.selectbox("Select Regression Algorithm", ('Random Forest', 'Linear Regression', 'K-Nearest Neighbor')) 
            if self.chosen_classifier == 'Random Forest': 
                self.n_trees = st.sidebar.slider('number of trees', 1, 1000, 1)
   
        elif self.type == "Classification":
            self.chosen_classifier = st.sidebar.selectbox("Select Classifier", ('Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbor')) 
            if self.chosen_classifier == 'Logistic Regression': 
                self.max_iter = st.sidebar.slider('max iterations', 1, 100, 10)
            elif self.chosen_classifier == 'K-Nearest Neighbor': 
                self.n_neighbors = st.sidebar.slider('value of k', 1, int((self.X_train.size)/2), 1)
           
       
    # Model training and predicitons 
    def predict(self, predict_btn):    

        if self.type == "Regression":    
            if self.chosen_classifier == 'Random Forest':
                self.alg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=self.n_trees)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
        
            elif self.chosen_classifier=='Linear Regression':
                self.alg = LinearRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier=='K-Nearest Neighbor':
                self.alg = KNeighborsRegressor(n_neighbors=3)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

               

        elif self.type == "Classification":
            if self.chosen_classifier == 'Logistic Regression':
                self.alg = LogisticRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
        
            elif self.chosen_classifier=='Naive Bayes':
                self.alg = GaussianNB()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
          
            elif self.chosen_classifier=='K-Nearest Neighbor':
                self.alg = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

           

        result = pd.DataFrame(columns=['Actual', 'Actual_Train', 'Prediction', 'Prediction_Train'])
        result_train = pd.DataFrame(columns=['Actual_Train', 'Prediction_Train'])
        result['Actual'] = self.y_test
        result_train['Actual_Train'] = self.y_train
        result['Prediction'] = self.predictions
        result_train['Prediction_Train'] = self.predictions_train
        result.sort_index()
        self.result = result
        self.result_train = result_train

        return self.predictions, self.predictions_train, self.result, self.result_train

    # Get the result metrics of the model
    def get_metrics(self):
        self.error_metrics = {}
        if self.type == 'Regression':
            self.error_metrics['MSE_test'] = mean_squared_error(self.y_test, self.predictions)
            self.error_metrics['MSE_train'] = mean_squared_error(self.y_train, self.predictions_train)
            return st.markdown('### MSE Train: ' + str(round(self.error_metrics['MSE_train'], 3)) + 
            ' -- MSE Test: ' + str(round(self.error_metrics['MSE_test'], 3)))

        elif self.type == 'Classification':
            self.error_metrics['Accuracy_test'] = accuracy_score(self.y_test, self.predictions)
            self.error_metrics['Accuracy_train'] = accuracy_score(self.y_train, self.predictions_train)
            return st.markdown('### Accuracy Train: ' + str(round(self.error_metrics['Accuracy_train'], 3)) +
            ' -- Accuracy Test: ' + str(round(self.error_metrics['Accuracy_test'], 3)))

    # Plot the predicted values and real values
    def plot_result(self):
        
        output_file("slider.html")

        s1 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
        s1.circle(self.result_train.index, self.result_train.Actual_Train, size=12, color="Black", alpha=1, legend_label = "Actual")
        s1.triangle(self.result_train.index, self.result_train.Prediction_Train, size=12, color="Red", alpha=1, legend_label = "Prediction")
        tab1 = Panel(child=s1, title="Train Data")

        if self.result.Actual is not None:
            s2 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
            s2.circle(self.result.index, self.result.Actual, size=12, color=Set3[5][3], alpha=1, legend_label = "Actual")
            s2.triangle(self.result.index, self.result.Prediction, size=12, color=Set3[5][4], alpha=1, legend_label = "Prediction")
            tab2 = Panel(child=s2, title="Test Data")
            tabs = Tabs(tabs=[ tab1, tab2 ])
        else:

            tabs = Tabs(tabs=[ tab1])

        st.bokeh_chart(tabs)

       
    # File selector module for web app
    def file_selector(self):
        file = st.sidebar.file_uploader("Upload CSV file", type="csv")
        if file is not None:
            data = pd.read_csv(file)
            return data
        else:
            st.text("Please upload a csv file")
        
    
    def print_table(self):
        if len(self.result) > 0:
            result = self.result[['Actual', 'Prediction']]
            st.dataframe(result.sort_values(by='Actual',ascending=False).style.highlight_max(axis=0))
    
    def set_features(self):
        self.features = st.sidebar.multiselect('Select all the variables that go into the model(features + target)', self.data.columns )

if __name__ == '__main__':
    controller = kwikml()
    try:
        controller.data = controller.file_selector()

        if controller.data is not None:
            if st.sidebar.checkbox('Display data'):
                st.subheader('Raw data')
                st.write(controller.data)
            if st.sidebar.checkbox('Display Null Values'):
                st.subheader('Null Values')
                all_null = controller.data.isnull().sum()
                st.dataframe(all_null)
                if st.sidebar.checkbox("Remove Null values"):
                    st.subheader('Dropping Null Values')
                    st.success("Dropped all Null values")
                    controller.data.dropna(inplace=True)
                    st.dataframe(controller.data)  
            split_data = st.sidebar.slider('Randomly reduce data size %', 1, 100, 10 )
            train_test = st.sidebar.slider('Train-test split %', 1, 99, 66 )
            controller.set_features()
        if len(controller.features) > 1:
            controller.prepare_data(split_data, train_test)
            controller.set_classifier_properties()
            predict_btn = st.sidebar.button('Predict')  
    except (AttributeError, ParserError, KeyError) as e:
        st.markdown('<span style="color:blue">Invalid File Type</span>', unsafe_allow_html=True)  


    if controller.data is not None and len(controller.features) > 1:
        if predict_btn:
            st.sidebar.text("Progress:")
            my_bar = st.sidebar.progress(0)
            predictions, predictions_train, result, result_train = controller.predict(predict_btn)
            for percent_complete in range(100):
                my_bar.progress(percent_complete + 1)
            
            controller.get_metrics()        
            controller.plot_result()
            controller.print_table()

      
   
    






