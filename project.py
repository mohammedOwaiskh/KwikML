import base64
import os
import sys
import time
from msilib.schema import Error
from re import S

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from bokeh.io import output_file, show
from bokeh.layouts import column, layout
from bokeh.models import BoxAnnotation, Panel, Tabs, Toggle
from bokeh.palettes import Set3
from bokeh.plotting import figure
from pandas.errors import ParserError
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.svm import SVC

st.set_page_config(page_title="KwikML", layout="wide", page_icon="⚡")
hide_streamlit_style = """
            <style>
			    #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("⚡ KwikML - A Machine Learning Workbench ⚡")
st.subheader(
    "Helping you to get familiar with machine learning models directly from your web browser"
)

st.markdown(
    """
    - 🗂️ Upload dataset
    - ⚙️ Select a model and set its hyper-parameters
    - 📉 Train it and check its performance metrics on train and test data
    - 🩺 Diagnose for low accuracy and experiment with other settings
    -----
    """
)


# Main kwikml class
class kwikml:

    model_imports = {
        "Logistic Regression": "from sklearn.linear_model import LogisticRegression",
        "Linear Regression": "from sklearn.linear_model import LinearRegression",
        "Random Forest": "from sklearn.ensemble import RandomForestRegressor",
        "K-Nearest Neighbor": "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor",
        "Naive Bayes": "from sklearn.naive_bayes import GaussianNB",
    }
    model_infos = {
        "Logistic Regression": """
        - A logistic regression is only suited to **linearly separable** problems
        - It's computationally fast and interpretable by design
        - It can handle non-linear datasets with appropriate feature engineering
    """,
        "Linear Regression": """
        - It tries to find out the best possible linear relationship between the input features and the target variable(y).
        - Its important you understand the relationship between your dependent variable and all the independent variables and whether they have a linear trend. Only then you can afford to use them in your model to get a good output.
    """,
        "Random Forest": """
        - They have lower risk of overfitting in comparison with decision trees
        - They are robust to outliers
        - They are computationally intensive on large datasets 
        - They are not easily interpretable
    """,
        "K-Nearest Neighbor": """
        - KNNs are intuitive and simple. They can also handle different metrics
        - KNNs don't build a model per se. They simply tag a new data based on the historical data
        - They become very slow as the dataset size grows
    """,
        "Naive Bayes": """
        - The Naive Bayes algorithm is very fast
        - It works well with high-dimensional data such as text classification problems
        - The assumption that all the features are independent is not always respected in real-life applications
    """,
    }

    # Preloaded Datasets
    def titanic_data(self):
        self.file = pd.read_csv("datasets/titanic_data.csv")
        return self.file

    def price_area_place(self):
        self.file = pd.read_csv("datasets/place_area_price.csv")
        return self.file

    def diabetes(self):
        self.file = pd.read_csv("datasets/diabetes.csv")
        return self.file

    # to automatically handle your data
    def prepare_data(self, split_data, train_test):
        # Reduce data size
        data = self.data[self.features]
        data = data.sample(frac=round(split_data / 100, 2))

        # converting into most frequent for categoricals
        cat_imp = SimpleImputer(strategy="most_frequent")
        if len(data.loc[:, data.dtypes == "object"].columns) != 0:
            data.loc[:, data.dtypes == "object"] = cat_imp.fit_transform(
                data.loc[:, data.dtypes == "object"]
            )
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        data.loc[:, data.dtypes != "object"] = imp.fit_transform(
            data.loc[:, data.dtypes != "object"]
        )

        # One hot encoding for categorical variables
        cats = data.dtypes == "object"
        le = LabelEncoder()
        for x in data.columns[cats]:
            sum(pd.isna(data[x]))
            data.loc[:, x] = le.fit_transform(data[x])
        onehotencoder = OneHotEncoder()
        data.loc[:, ~cats].join(
            pd.DataFrame(
                data=onehotencoder.fit_transform(data.loc[:, cats]).toarray(),
                columns=onehotencoder.get_feature_names_out(),
            )
        )

        # Set target column
        target_options = data.columns
        self.chosen_target = st.selectbox("Select target column", (target_options))

        # Standardize the feature data
        X = data.loc[:, data.columns != self.chosen_target]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = data.loc[:, data.columns != self.chosen_target].columns
        y = data[self.chosen_target]

        # Train test split
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=(1 - train_test / 100), random_state=42
            )
        except Exception as e:
            print(e)
            st.markdown(
                '<span style="color:red">With this amount of data and split size the train data will have no records, <br /> Please change reduce and split parameter <br /> </span>',
                unsafe_allow_html=True,
            )

    # Classifier type and algorithm selection
    def set_classifier_properties(self):
        self.type = st.selectbox("Algorithm type", ("Classification", "Regression"))
        if self.type == "Regression":
            self.chosen_classifier = st.selectbox(
                "Select Regression Algorithm",
                ("Random Forest", "Linear Regression", "K-Nearest Neighbor"),
            )
            if self.chosen_classifier == "Random Forest":
                self.n_trees = st.slider("number of trees", 1, 1000, 1)

        elif self.type == "Classification":
            self.chosen_classifier = st.selectbox(
                "Select Classifier",
                ("Logistic Regression", "Naive Bayes", "K-Nearest Neighbor"),
            )
            if self.chosen_classifier == "Logistic Regression":
                self.max_iter = st.slider("max iterations", 1, 100, 10)
            elif self.chosen_classifier == "K-Nearest Neighbor":
                self.n_neighbors = st.slider("Value of k", 1, 20, 1)

    # Model training and predicitons
    def predict(self):
        if self.type == "Regression":
            if self.chosen_classifier == "Random Forest":
                self.alg = RandomForestRegressor(
                    max_depth=2, random_state=0, n_estimators=self.n_trees
                )
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier == "Linear Regression":
                self.alg = LinearRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier == "K-Nearest Neighbor":
                self.alg = KNeighborsRegressor(n_neighbors=3)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

        elif self.type == "Classification":
            if self.chosen_classifier == "Logistic Regression":
                self.alg = LogisticRegression()
                self.model = self.alg.fit(self.X_train.values, self.y_train.values)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train.values)
                self.predictions = predictions

            elif self.chosen_classifier == "Naive Bayes":
                self.alg = GaussianNB()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier == "K-Nearest Neighbor":
                self.alg = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

        result = pd.DataFrame(
            columns=["Actual", "Actual_Train", "Prediction", "Prediction_Train"]
        )
        result_train = pd.DataFrame(columns=["Actual_Train", "Prediction_Train"])
        result["Actual"] = self.y_test
        result_train["Actual_Train"] = self.y_train
        result["Prediction"] = self.predictions
        result_train["Prediction_Train"] = self.predictions_train
        result.sort_index()
        self.result = result
        self.result_train = result_train

        return self.predictions, self.predictions_train, self.result, self.result_train

    # Get the result metrics of the model
    def get_metrics(self):
        self.error_metrics = {}
        if self.type == "Regression":
            self.error_metrics["MSE_test"] = mean_squared_error(
                self.y_test, self.predictions
            )
            self.error_metrics["MSE_train"] = mean_squared_error(
                self.y_train, self.predictions_train
            )
            return st.markdown(
                "### MSE Train: "
                + str(round(self.error_metrics["MSE_train"], 3))
                + " -- MSE Test Score: "
                + str(round(self.error_metrics["MSE_test"], 3))
            )

        elif self.type == "Classification":
            self.error_metrics["Accuracy_test"] = accuracy_score(
                self.y_test, self.predictions
            )
            self.error_metrics["Accuracy_train"] = accuracy_score(
                self.y_train, self.predictions_train
            )
            return st.markdown(
                "### Accuracy Train: "
                + str(round(self.error_metrics["Accuracy_train"], 3))
                + " -- Accuracy Test: "
                + str(round(self.error_metrics["Accuracy_test"], 3))
            )

    # Plot the predicted values and real values
    def plot_result(self):

        output_file("slider.html")

        s1 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
        s1.circle(
            self.result_train.index,
            self.result_train.Actual_Train,
            size=12,
            color="Black",
            alpha=1,
            legend_label="Actual",
        )
        s1.triangle(
            self.result_train.index,
            self.result_train.Prediction_Train,
            size=12,
            color="Red",
            alpha=1,
            legend_label="Prediction",
        )
        tab1 = Panel(child=s1, title="Train Data")

        if self.result.Actual is not None:
            s2 = figure(
                plot_width=800, plot_height=500, background_fill_color="#5f27cd"
            )
            s2.circle(
                self.result.index,
                self.result.Actual,
                size=12,
                color=Set3[5][3],
                alpha=1,
                legend_label="Actual",
            )
            s2.triangle(
                self.result.index,
                self.result.Prediction,
                size=12,
                color=Set3[5][4],
                alpha=1,
                legend_label="Prediction",
            )
            tab2 = Panel(child=s2, title="Test Data")
            tabs = Tabs(tabs=[tab1, tab2])
        else:

            tabs = Tabs(tabs=[tab1])

        st.bokeh_chart(tabs)

    # File selector module for web app
    def file_selector(self):
        dataset_import = st.sidebar.selectbox(
            "Where do you want to import data set from?",
            ["Select Import type", "Pre-Loaded", "Upload"],
        )
        if dataset_import == "Pre-Loaded":
            dataset = st.sidebar.selectbox(
                "Select Dataset",
                [
                    "Select a Dataset",
                    "Titanic Data",
                    "Place, Area & Price",
                    "Diabetes Dataset",
                ],
            )
            if dataset == "Titanic Data":
                self.data = self.titanic_data()
                self.filename = "titanic_data.csv"
                return self.data
            elif dataset == "Place, Area & Price":
                self.data = self.price_area_place()
                self.filename = "place_area_price.csv"
                return self.data
            elif dataset == "Diabetes Dataset":
                self.data = self.diabetes()
                self.filename = "diabetes.csv"
                return self.data
        elif dataset_import == "Upload":
            file = st.sidebar.file_uploader("Upload CSV file", type="csv")
            # print(file.name)
            if file is not None:
                data = pd.read_csv(file)
                self.filename = file.name
                return data
            else:
                st.text("Please upload a csv file")

    def print_table(self):
        if len(self.result) > 0:
            result = self.result[["Actual", "Prediction"]]
            st.dataframe(
                result.sort_values(by="Actual", ascending=False).style.highlight_max(
                    color="#FF4B4B", axis=0
                )
            )

    def set_features(self):
        self.features = st.multiselect(
            "Select all the variables that go into the model(features + target)",
            self.data.columns,
        )

    def get_code_snippet(self):
        st.header("** Try it yourself!🖊️ **")
        return f"""
        ----------------------------------------------------------
            import numpy as np
            import pandas as pd
            {self.model_imports[self.chosen_classifier]}
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split

            # import data
            file = "{self.filename}"
            # Tip: Make sure your dataset file is saved in the same folder as this code file
            data = pd.read_csv(file)
            df = pd.DataFrame(data)


            features = data.drop("{self.chosen_target}", axis='columns')
            target = data['{self.chosen_target}']
            
            x_train, x_test, y_train, y_test = train_test_split(features, target)

            model = {self.alg}
            model.fit(x_train, y_train)
            
            train_score = model.score(x_train, y_train)
            test_score = model.score(x_test, y_test) 

            print("train score =", train_score)
            print("test score =", test_score) 

        """


if __name__ == "__main__":
    controller = kwikml()

    try:
        controller.data = controller.file_selector()

        if controller.data is not None:

            with st.sidebar.expander("Data Pre-processing", True):
                raw_data = st.checkbox("Display data")
                null_val = st.checkbox("Display Null Values")
                drop_null = st.checkbox("Drop Null Values")
                shape = st.checkbox("Show Shape")
                sh_clm = st.checkbox("Show Columns")
                smry = st.checkbox("Summary")
            if raw_data:
                st.subheader("Raw data")
                st.write(controller.data)
            if null_val:
                st.subheader("Null Values")
                all_null = controller.data.isnull().sum()
                st.dataframe(all_null)
                if drop_null:
                    st.subheader("Dropping Null Values")
                    st.success("Dropped all Null values")
                    controller.data.dropna(inplace=True)
                    st.dataframe(controller.data)
            if shape:
                st.subheader("Shape")
                st.write(controller.data.shape)
            if sh_clm:
                st.subheader("Columns")
                all_columns = controller.data.columns.to_list()
                st.write(all_columns)
            if smry:
                st.subheader("Summary")
                st.write(controller.data.describe())

            # Data Visualization
            with st.sidebar.expander("Data Visualization"):
                type_of_plot = st.selectbox(
                    "Select Type of Plot", ["", "Area", "Bar", "Line", "Pie"]
                )
                df = pd.DataFrame(controller.data)
                all_columns_names = controller.data.columns.tolist()
            try:
                if type_of_plot == "Bar":
                    x_column = st.selectbox("Select x-axis", all_columns_names)
                    y_column = st.selectbox("Select y-axis", all_columns_names)
                    if want_query := st.checkbox("Enable Query column"):
                        query = st.selectbox("Select a query", all_columns_names)
                        all_values = controller.data.loc[:, query].tolist()
                        value = st.selectbox(
                            f"Select the value from {query}", all_values
                        )
                        df = df.query(f"{query} == {value}")
                    if generate_plot := st.button("Generate Plot"):
                        fig = px.bar(df, x=x_column, y=y_column)
                        st.plotly_chart(fig, True)
                elif type_of_plot == "Line":
                    x_column = st.selectbox("Select x-axis", all_columns_names)
                    y_column = st.selectbox("Select y-axis", all_columns_names)
                    if want_query := st.checkbox("Enable Query column"):
                        query = st.selectbox("Select a query", all_columns_names)
                        all_values = controller.data.loc[:, query].tolist()
                        value = st.selectbox(
                            f"Select the value from {query}", all_values
                        )
                        df = df.query(f"{query} == {value}")
                    fig = px.line(df, x=x_column, y=y_column)
                    if generate_plot := st.button("Generate Plot"):
                        st.plotly_chart(fig, True)

                elif type_of_plot == "Pie":
                    names = st.selectbox("Select Name:", all_columns_names)
                    values = st.selectbox("Select Value:", all_columns_names)
                    if generate_plot := st.button("Generate Plot"):
                        fig = px.pie(df, values=values, names=names, hole=0.3)
                        st.plotly_chart(fig, True)

                elif type_of_plot == "Area":
                    x_column = st.selectbox("Select data on x-axis", all_columns_names)
                    y_column = st.selectbox("Select data on y-axis", all_columns_names)
                    fig = px.area(df, x=x_column, y=y_column)
                    if generate_plot := st.button("Generate Plot"):
                        st.plotly_chart(fig, True)

            except Exception as e:
                st.warning(
                    "Incompatible column selected. Try selecting other columns or To know more about the data click on 'Display Data'."
                )

            # Model Generation
            with st.sidebar.expander("Model Generation"):
                split_data = st.slider("Randomly reduce data size %", 1, 100, 70)
                train_test = st.slider("Train-test split %", 1, 99, 66)
                controller.set_features()
                if len(controller.features) > 1:
                    controller.prepare_data(split_data, train_test)
                    controller.set_classifier_properties()
                predict_btn = st.button("Predict")

    except (AttributeError, ParserError, KeyError) as e:
        st.markdown("")

    if controller.data is not None and len(controller.features) > 1 and predict_btn:
        st.sidebar.text("Progress:")
        my_bar = st.sidebar.progress(0)
        predictions, predictions_train, result, result_train = controller.predict()
        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1)

        # Getting Accurary Reports
        st.header("Accuracy Report")
        controller.get_metrics()
        controller.plot_result()
        controller.print_table()
        c1, c2 = st.columns((1, 1))
        with c2:
            snippet = controller.get_code_snippet()
            st.code(snippet)
        with c1:
            st.header(f"**Tips on the {controller.chosen_classifier} 💡 **")
            st.info(controller.model_infos[controller.chosen_classifier])
