from flask import Flask, app, render_template, request
from final_model import *

import requests
import json
app = Flask(__name__)

def func(seq):
    #mediator between model.py and html 
    results = model_test(seq)
    return results 

@app.route("/", methods = ['POST','GET'])
def index():
    if request.method == 'GET':
        return render_template("index.html",result_azm="pending",result_cip="pending",result_cfx="pending")
    if request.method == 'POST':
        seq = request.form['seq']
        results = func(seq)
        return render_template("index.html",result_azm=results[0], result_cip=results[1], result_cfx=results[2])  

app.run(host="0.0.0.0", port = 500 )