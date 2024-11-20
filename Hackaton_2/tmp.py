import pandas as pd

import numpy as np

import warnings

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import iplot

from plotly.subplots import make_subplots

import sklearn.preprocessing as skp

def scale_dataset(df):
    column_names = df.columns
    column_names_without_diabetes = column_names.drop('Diabetes')
    scale_df = skp.StandardScaler().fit_transform(df[column_names_without_diabetes])
    df[column_names_without_diabetes] = pd.DataFrame(scale_df, columns=column_names_without_diabetes)
    return df

def plot_correlation_matrix(data):
    X = [label for label in data]
    N = data.shape[1]
    corr = data.corr().values
    hovertext = [[f'corr({X[i]}, {X[j]})= {corr[i][j]:.2f}' for j in range(N)] for i in range(N)]
    sns_colorscale = [[0.0, '#3f7f93'], [0.071, '#5890a1'], [0.143, '#72a1b0'], [0.214, '#8cb3bf'], [0.286, '#a7c5cf'], [0.357, '#c0d6dd'], [0.429, '#dae8ec'], [0.5, '#f2f2f2'], [0.571, '#f7d7d9'], [0.643, '#f2bcc0'], [0.714, '#eda3a9'], [0.786, '#e8888f'], [0.857, '#e36e76'], [0.929, '#de535e'], [1.0, '#d93a46']]
    heat = go.Heatmap(z=data.corr(), x=X, y=X, zmin=-1, zmax=1, xgap=1, ygap=1, colorscale=sns_colorscale, colorbar_thickness=20, colorbar_ticklen=3, hovertext=hovertext, hoverinfo='text')
    title = 'Correlation Matrix'
    layout = go.Layout(title_text=title, title_x=0.5, width=600, height=600, xaxis_showgrid=False, yaxis_showgrid=False, yaxis_autorange='reversed')
    fig = go.Figure(data=[heat], layout=layout)
    fig.show()

def sort_features(corr_matrix):
    series = corr_matrix['Diabetes'].drop('Diabetes').abs()
    return series.sort_values(ascending=False).to_list()

from sklearn.linear_model import LinearRegression

def linear_regressor(X_train, y_train, threshold=0.5):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    return lambda X_test: np.where(linreg.predict(X_test) > threshold, 1, 0)

from sklearn.linear_model import LogisticRegression

def logistic_regressor(X_train, y_train, threshold=0.5):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return lambda X_test: np.where(logreg.predict(X_test) > threshold, 1, 0)

from sklearn.neighbors import KNeighborsRegressor

def knn_regressor(X_train, y_train, threshold=0.5, n_neighbors=10):
    knnreg = KNeighborsRegressor(n_neighbors)
    knnreg.fit(X_train, y_train)
    return lambda X_test: np.where(knnreg.predict(X_test) > threshold, 1, 0)

from sklearn.metrics import precision_score

def precision(y_test, y_pred):
    return precision_score(y_test, y_pred)

from sklearn.metrics import recall_score

def recall(y_test, y_pred):
    return recall_score(y_test, y_pred)

from sklearn.metrics import f1_score

def f1_score(y_test, y_pred):
    return f1_score(y_test, y_pred)

def validation(regressor, X_test, y_test):
    y_pred = regressor(X_test)
    return (recall(y_test, y_pred), precision(y_test, y_pred), f1_score(y_test, y_pred))

