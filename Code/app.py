from flask import Flask, app, render_template
from final_model import *

import requests
import json
app = Flask(__name__)

def func():
    #mediator between model.py and html 
    results = model_test()
    return results 

@app.route("/")
def index():
    results = func()
    #result_azm = results[0]
    #result_cip = results[1]
    #result_cfx = results[2]
    return render_template("index.html",result_azm=results[0], result_cip=results[1], result_cfx=results[2])  

app.run(host="0.0.0.0", port = 500 )