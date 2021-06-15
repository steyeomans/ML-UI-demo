"""
Simple flask application to demonstrate web application ability to interact
with databases and run machine learning operations.
"""

from flask import (Flask, request, make_response, json, render_template, 
    redirect, url_for, jsonify, Response, flash)
import numpy as np
import os
#import connections
#import models
import forms
from dotenv import load_dotenv

load_dotenv()

application = Flask(__name__)
application.secret_key = os.environ['FLASK_KEY']

data = 1 #models.Prep_data()
std_scaler = 1 #data.std_scale()
ols = 1 #models.do_OLS(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)
svr = 1 #models.do_SVR(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)
rfr = 1 #models.do_RFR(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)
mlp = 1 #models.do_MLP(data.X_train_scaled, data.y_train, data.X_test_scaled, data.y_test)

@application.route('/', methods=('GET','POST'))
def home():
    #lgform = forms.LoginForm()
    #if request.method == 'POST':
    #    redir = LoginModal(lgform)
    #    if redir:
    #        return redirect(redir)
    return render_template('home.html',) #LoginForm = lgform, SLACK_APP_ID=SLACK_APP_ID)

@application.route('/GetPrediction', methods=('GET','POST'))
def get_pred():
    #lgform = forms.LoginForm()
    #if request.method == 'POST':
    #    redir = LoginModal(lgform)
    #    if redir:
    #        return redirect(redir)
    input_form = forms.MLPredForm()    
    if request.method == 'POST' and input_form.validate():
        new_data = [input_form.med_inc.data/10000, input_form.avg_house_age.data, 
            input_form.avg_rooms.data, input_form.avg_bedrooms.data,
            input_form.population.data/1000, input_form.avg_occupancy.data
        ]
        new_data = np.reshape(new_data,(1,-1))
        new_scaled_data = std_scaler.transform(new_data)
        pred_dict = {}
        pred_dict['Linear Regression'] = [ols.predict(new_scaled_data)*100000, ols.test_score]
        pred_dict['Support Vector Regression'] = [svr.predict(new_scaled_data)*100000, svr.test_score]
        pred_dict['Random Forest Regression'] = [rfr.predict(new_scaled_data)*100000, rfr.test_score]
        pred_dict['Multi Layer Perceptron Regression'] = [mlp.predict(new_scaled_data)*100000, mlp.test_score]
        return render_template('GetPredResults.html', pred_dict=pred_dict) #LoginForm = lgform, SLACK_APP_ID=SLACK_APP_ID)
    elif request.method == 'GET':
        prefill_data = data.samples_df.sample().to_dict(orient='records')[0]
        input_form.med_inc.data = int(prefill_data['MedInc']*10000)
        input_form.avg_house_age.data = int(prefill_data['HouseAge'])
        input_form.avg_rooms.data = prefill_data['AveRooms']
        input_form.avg_bedrooms.data = prefill_data['AveBedrms']
        input_form.population.data = int(prefill_data['Population']*1000)
        input_form.avg_occupancy.data = prefill_data['AveOccup']
    #return errors if validation does not pass
    return render_template('GetPredEntry.html', input_form=input_form) #LoginForm = lgform, SLACK_APP_ID=SLACK_APP_ID)

@application.route('/TestModels', methods=('GET','POST'))
def test_pred():
    #lgform = forms.LoginForm()
    #if request.method == 'POST':
    #    redir = LoginModal(lgform)
    #    if redir:
    #        return redirect(redir)
    input_form = forms.MLPredForm()    
    if request.method == 'POST' and input_form.validate():
        new_data = [input_form.med_inc.data/10000, input_form.avg_house_age.data, 
            input_form.avg_rooms.data, input_form.avg_bedrooms.data,
            input_form.population.data/1000, input_form.avg_occupancy.data
        ]
        real_value = input_form.prediction.data
        new_data = np.reshape(new_data,(1,-1))
        new_scaled_data = std_scaler.transform(new_data)
        pred_dict = {}
        pred_dict['Linear Regression'] = [ols.predict(new_scaled_data)*100000, ols.test_score, 
            int((ols.predict(new_scaled_data)*100000)-real_value)]
        pred_dict['Support Vector Regression'] = [svr.predict(new_scaled_data)*100000, svr.test_score,
            int(svr.predict(new_scaled_data)*100000-real_value)]
        pred_dict['Random Forest Regression'] = [rfr.predict(new_scaled_data)*100000, rfr.test_score,
            int(rfr.predict(new_scaled_data)*100000-real_value)]
        pred_dict['Multi Layer Perceptron Regression'] = [mlp.predict(new_scaled_data)*100000, mlp.test_score,
            int(mlp.predict(new_scaled_data)*100000-real_value)]
        return render_template('TestPredResults.html', pred_dict=pred_dict, 
            entered_pred=real_value) #LoginForm = lgform, SLACK_APP_ID=SLACK_APP_ID)
    elif request.method == 'GET':
        prefill_data = data.samples_df.sample().to_dict(orient='records')[0]
        input_form.med_inc.data = int(prefill_data['MedInc']*10000)
        input_form.avg_house_age.data = int(prefill_data['HouseAge'])
        input_form.avg_rooms.data = prefill_data['AveRooms']
        input_form.avg_bedrooms.data = prefill_data['AveBedrms']
        input_form.population.data = int(prefill_data['Population']*1000)
        input_form.avg_occupancy.data = prefill_data['AveOccup']
        input_form.prediction.data = int(prefill_data['MedHouseVal']*100000)
    #return errors if validation does not pass
    return render_template('TestPredEntry.html', input_form=input_form) #LoginForm = lgform, SLACK_APP_ID=SLACK_APP_ID)

#if __name__ == '__main__':
#    application.run(debug=True)
