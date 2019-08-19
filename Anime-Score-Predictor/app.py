#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:12:13 2019

@author: siggy
"""

from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import predictanime as pa

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)




@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        title = request.form.get('title')
        genre = request.form.get('genre')
        platform = request.form.get('platform')
        summary = request.form.get('summary')
        predictor = pa.AnimePredictor()
        result    = predictor.get_score(title, [genre], platform, summary)
        
    
        return render_template('prediction.html', result="This anime will be {}".format(result))
    
    return render_template('index.html')

@app.route('/about', methods=['POST', 'GET'])     
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug="True")