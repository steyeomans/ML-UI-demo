"""
Data connection and transformation functions for the app.
"""

import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

class Prep_data:
    def __init__(self, test_size=0.2):
        data_df = datasets.fetch_california_housing(as_frame=True)['frame']
        #not a template on feature engineering best practices, stick with the easy ones
        self.X = data_df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
            'Population', 'AveOccup']].copy()
        self.y = data_df[['MedHouseVal']].values.ravel()
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42)

        self.samples_df = data_df.sample(20)
    
    def std_scale(self):
        std_scaler = StandardScaler().fit(self.X_train)
        self.X_train_scaled = std_scaler.transform(self.X_train)
        self.X_test_scaled = std_scaler.transform(self.X_test)
        return std_scaler
    
def do_OLS(X, y, X_test, y_test):
    ols = LinearRegression()
    ols.fit(X, y)
    ols.test_score = ols.score(X_test, y_test)
    print('OLS score:', ols.test_score)
    return ols
    
#def do_SGD(X, y, X_test, y_test):
    #params = {
    #    'alpha':[0.1,0.01,0.001,0.0001,0.00001],
    #    'learning_rate':['constant','optimal','invscaling','adaptive'],
    #    'verbose':[3,],
    #    'random_state':[42,],
    #}
    #sgd = GridSearchCV(estimator=SGDRegressor(), param_grid=params, scoring='r2')
    ##sgd = SGDRegressor(random_state=42)
    #sgd.fit(X,y)
    #print(sgd.cv_results_)
    #print(sgd.best_params_)
    #print('SGD score:', sgd.score(X_test, y_test))
    #return sgd

def do_SVR(X, y, X_test, y_test):
    #params = [
    #    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    #    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    #]
    #svr = GridSearchCV(estimator=SVR(), param_grid=params, scoring='r2')
    svr = SVR(C=100)
    svr.fit(X, y)
    svr.test_score = svr.score(X_test, y_test)
    print('SVR score:', svr.test_score)
    return svr

def do_RFR(X, y, X_test, y_test):
    #params = {
    #    'n_estimators':[10,100,1000],
    #    'max_depth':[2,3,4,5,6,7,8],
    #    'n_jobs':[4,],
    #    'verbose':[3,],
    #}
    #rfr = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, scoring='r2')
    rfr = RandomForestRegressor(max_depth=8, random_state=42)
    rfr.fit(X,y)
    #print(rfr.cv_results_)
    rfr.test_score = rfr.score(X_test, y_test)
    print('RFR score:', rfr.test_score)
    return rfr

def do_MLP(X, y, X_test, y_test):
    #params = {
    #    'hidden_layer_sizes':[(10,10,10),4*(10,),5*(10,),6*(10,),7*(10,),8*(10,),9*(10,),(100,),(10,10,),(10,10,10,)],
    #    'activation':['relu','identity'],
    #    'solver':['sgd','adam'],
    #    'random_state':[42,],
    #    'verbose':[3,],
#
    #}
    #mlp = GridSearchCV(estimator=MLPRegressor(), param_grid=params, scoring='r2')
    mlp = MLPRegressor(hidden_layer_sizes=7*(10,), random_state=42)
    mlp.fit(X,y)
    #print(mlp.cv_results_)
    #print(mlp.best_params_)
    mlp.test_score = mlp.score(X_test, y_test)
    print('MLP score:', mlp.test_score)
    return mlp

if __name__ == '__main__':
    data = Prep_data()
    data.std_scale()
    do_OLS(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)
    #do_SGD(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)
    do_SVR(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)
    do_RFR(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)
    do_MLP(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)