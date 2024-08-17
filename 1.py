import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

# Function to load dataset
def load_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.warning("You need to upload a CSV file.")
        return None

# Function to clean data
def clean_data(data):
    st.subheader("Data Cleaning")
    st.write("Handling missing values...")
    data = data.dropna()  # Simple dropna, more advanced methods can be implemented
    st.write(f"Data shape after removing missing values: {data.shape}")
    
    st.write("Removing duplicates...")
    data = data.drop_duplicates()
    st.write(f"Data shape after removing duplicates: {data.shape}")
    
    return data

# Function to display data visualization
def visualize_data(data, task):
    st.subheader("Data Visualization")

    target_column = st.selectbox("Select Target Column", data.columns)
    
    if task == "Classification":
        st.write("Distribution of Target Variable:")
        if data[target_column].dtype == 'object':
            st.bar_chart(data[target_column].value_counts())
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data[target_column], bins=20, color='skyblue', edgecolor='black')
            ax.set_xlabel(target_column)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {target_column}')
            st.pyplot(fig)

    st.write("Correlation Heatmap:")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("Pair Plot:")
    fig = sns.pairplot(data)
    st.pyplot(fig.fig)

    st.write("Box Plot:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, orient="h", ax=ax)
    st.pyplot(fig)

# Function to do feature selection
def feature_selection(X, y, task):
    st.subheader("Feature Selection")
    
    if task == "Classification":
        st.write("Selecting top features based on ANOVA F-test...")
        selector = SelectKBest(f_classif, k='all')
    else:
        st.write("Selecting top features based on Mutual Information Regression...")
        selector = SelectKBest(mutual_info_regression, k='all')
    
    selector.fit(X, y)
    scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
    scores = scores.sort_values(by='Score', ascending=False)

    st.write(scores)

    num_features = st.slider("Select number of top features to keep:", min_value=1, max_value=len(X.columns), value=5)
    selected_features = scores['Feature'].iloc[:num_features].values

    return X[selected_features]

# Function to perform feature extraction (PCA)
def feature_extraction(X):
    st.subheader("Feature Extraction (PCA)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    st.write("Explained variance ratio by each component:")
    st.write(pca.explained_variance_ratio_)

    st.write("PCA Plot:")
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c='blue')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('PCA Plot')
    st.pyplot(fig)

    return X_pca

# Function to evaluate a single model
def evaluate_model(name, model, X_train, X_test, y_train, y_test, task):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == "Classification":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        return {
            "Model": name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        }
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        return {
            "Model": name,
            "RMSE": rmse,
            "R² Score": r2
        }

# Function to evaluate multiple models with default parameters
def evaluate_models(X_train, X_test, y_train, y_test, task):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    model_scores = []
    for model_name, model in models.items():
        score = evaluate_model(model_name + " (Default Params)", model, X_train, X_test, y_train, y_test, task)
        model_scores.append(score)

    return model_scores

# Suggest and evaluate models
def suggest_model(X, y, task):
    st.subheader("Model Suggestion, Evaluation, and Hyperparameter Tuning")

    if task == "Classification":
        metric = st.selectbox("Select the primary metric for model comparison", ["F1 Score", "Accuracy"])
        scoring = 'f1_weighted' if metric == "F1 Score" else 'accuracy'

        models = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        params = {
            "Logistic Regression": {'C': [0.1, 1, 10]},
            "Support Vector Machine": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            "Random Forest": {'n_estimators': [100, 200], 'max_depth': [5, 10]},
            "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
            "Decision Tree": {'max_depth': [5, 10, 15]},
            "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
        }

    else:
        metric = st.selectbox("Select the primary metric for model comparison", ["RMSE", "R² Score"])
        scoring = 'neg_root_mean_squared_error' if metric == "RMSE" else 'r2'

        models = {
            "Linear Regression": LinearRegression(),
            "Support Vector Regressor": SVR(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBoost Regressor": XGBRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor()
        }

        params = {
            "Linear Regression": {},
            "Support Vector Regressor": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            "Random Forest Regressor": {'n_estimators': [100, 200], 'max_depth': [5, 10]},
            "XGBoost Regressor": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
            "Decision Tree Regressor": {'max_depth': [5, 10, 15]},
            "AdaBoost Regressor": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            "Gradient Boosting Regressor": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_scores = []

    # Evaluate models with default parameters
    st.write("Evaluating models with default parameters...")
    model_scores.extend(evaluate_models(X_train, X_test, y_train, y_test, task))

    # Evaluate models with hyperparameter tuning
    st.write("Evaluating models with hyperparameter tuning...")
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, params[model_name], cv=5, n_jobs=-1, scoring=scoring)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        st.write(f"Best parameters for {model_name}: {best_params}")
        model = grid_search.best_estimator_
        score = evaluate_model(model_name + " (Tuned Params)", model, X_train, X_test, y_train, y_test, task)
        model_scores.append(score)

    # Perform feature extraction and selection, then evaluate models with hyperparameter tuning
    st.write("Performing feature extraction and selection, then evaluating models with hyperparameter tuning...")
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, params[model_name], cv=5, n_jobs=-1, scoring=scoring)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        st.write(f"Best parameters for {model_name}: {best_params}")
        model = grid_search.best_estimator_
        score = evaluate_model(model_name + " (Tuned Params, With Feature Selection)", model, X_train, X_test, y_train, y_test, task)
        model_scores.append(score)

    # Display the scores in a table format
    st.subheader("Model Performance Comparison")
    st.write(pd.DataFrame(model_scores).set_index('Model'))

    # Suggest the best model based on the selected metric
    best_model_row = max(model_scores, key=lambda x: x[metric] if task == "Classification" else -x[metric])
    st.write(f"Suggested Model: **{best_model_row['Model']}** with a {metric} of {best_model_row[metric]:.2f}")

    if st.checkbox("Show all model results"):
        st.write(f"All model comparisons are shown using the selected metric: {metric}")
        st.write(pd.DataFrame(model_scores).set_index('Model'))

from sklearn.preprocessing import LabelEncoder

# Function to convert categorical features to numerical
def convert_categorical_to_numerical(data):
    st.subheader("Converting Categorical to Numerical")
    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        st.write(f"Categorical columns detected: {list(categorical_columns)}")
        encoding_type = st.selectbox("Choose encoding type", ["Label Encoding", "One-Hot Encoding"])

        if encoding_type == "Label Encoding":
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le
            st.write("Label Encoding applied to categorical columns.")
        
        elif encoding_type == "One-Hot Encoding":
            data = pd.get_dummies(data, columns=categorical_columns)
            st.write("One-Hot Encoding applied to categorical columns.")
        
    else:
        st.write("No categorical columns detected.")
    
    return data

def main():
    st.title("Comprehensive ML Modeling App")

    task = st.selectbox("Select Task", ("Classification", "Regression"))

    data = load_data()
    if data is not None:
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Data Cleaning
        data = clean_data(data)

        # Convert categorical columns to numerical
        data = convert_categorical_to_numerical(data)

        # EDA and Visualization
        visualize_data(data, task)

        target_column = st.selectbox("Select Target Column for Model Training", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Feature Selection
        X_selected = feature_selection(X, y, task)

        # Feature Extraction (PCA)
        X_pca = feature_extraction(X_selected)

        # Model Suggestion, Evaluation, and Hyperparameter Tuning
        if st.button("Suggest Model"):
            suggest_model(X, y, task)  # Evaluate models with different configurations

if __name__ == '__main__':
    main()
