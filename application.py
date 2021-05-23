"""
Simple flask application to demonstrate web application ability to interact
with databases and run machine learning operations.
"""

from flask import (Flask, request, make_response, json, render_template, 
    redirect, url_for, jsonify, Response, flash)
import os
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ['FLASK_KEY']

@app.route('/', methods=('GET','POST'))
def home():
    #lgform = forms.LoginForm()
    #if request.method == 'POST':
    #    redir = LoginModal(lgform)
    #    if redir:
    #        return redirect(redir)
    return render_template('home.html',) #LoginForm = lgform, SLACK_APP_ID=SLACK_APP_ID)

if __name__ == '__main__':
    app.run(debug=True)
