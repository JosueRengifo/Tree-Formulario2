#para el servicio web
from flask import Flask,jsonify,render_template,url_for
from clasificador import Clasificador
import numpy as np
app = Flask(__name__)
@app.route('/api/<params>')
def Clasificator(params):
    rRaw=str.split(params,',')
    feature=np.array([rRaw])
    classification=Clasificador(feature)
    return jsonify(
        feature=params,
        classification=str(classification[0])
        )

@app.route("/")
def hello(name=None):
    return render_template('index.html', name=name)