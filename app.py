from flask import Flask, render_template, url_for, request,redirect
import model
from bs4 import BeautifulSoup
import requests
import numpy as np
import json
from tqdm import tqdm
import seaborn as sns
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from google_play_scraper import Sort, reviews, app
#from keras.models import load_model
#from pycaret.classification import *

import os
#os.chdir('C:/Users/joshi_f9n026d/PycharmProjects/fraud_mp/lr-model.pkl')
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

#sns.set(style='whitegrid', palette='muted', font_scale=1.2)


appy = Flask(__name__)



@appy.route('/', methods=['GET', 'POST'])
def home():
    if request.method=='POST':
        return redirect(url_for('predict'))
    else:
        return render_template('home.html')


@appy.route('/predict', methods=['POST'])
def predict():
    score = 0

    if request.method == 'POST':
        app_packages =[]
        data = {'Ad Supported': "", 'In App Purchases': '', 'Rating': "", 'review': ''}
        url = request.form['urle']
        req = requests.get(url)

        content = req.text
        soup = BeautifulSoup(content)


        if soup.find("div",class_="BHMmbe") is None:
            data['Rating']=0
        else:
            data['Rating'] = soup.find("div",class_="BHMmbe").text.strip()
        url = url.split("id=")
        app_packages.append(url[1])
        print(app_packages)
        app_infos = []

        for ap in tqdm(app_packages):
            try:
                info = app(ap,lang='en',country='us')
            except Exception as e:
                print(e)
            else:
                del info['comments']
                app_infos.append(info)
        print(app_infos)
        def print_json(json_object):
            json_str = json.dumps(
                json_object,
                indent=2,
                sort_keys=True,
                default=str
            )
            data['Ad Supported']=app_infos[0]['containsAds']
            if app_infos[0]['inAppProductPrice'] != 'null':
                data['In App Purchases'] = True
            else:
                data['In App Purchases'] = False


            data['review'] = app_infos[0]['score']
            print(highlight(json_str, JsonLexer(), TerminalFormatter()))

        print_json(app_infos[0])

        #new = pd.DataFrame([data])
        #new_model = load_model('lr-model')
       # new = pd.DataFrame.from_records(new)
        #result = new_model.predict(new)
        variety = model.classify(int(data['Ad Supported']), int(data['In App Purchases']), data['Rating'], data['review'])
        score = variety
        print(score)
        # result=new_model.predict_classes(data)
        # print(len(result))
        return render_template('result.html', score=score,icon= app_infos[0]['icon'])
    return render_template('result.html',score=score)

if __name__ == '__main__':
    appy.debug = True
    appy.run()