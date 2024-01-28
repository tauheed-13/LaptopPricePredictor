from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    size = float(request.form.get('size'))
    type_name = request.form.get('type_name')
    ram = int(request.form.get('ram'))
    gpu = request.form.get('gpu')
    os = request.form.get('os')
    weight = float(request.form.get('weight'))
    touchscreen = request.form.get('touchscreen')
    ips = request.form.get('ips')
    ScreenResolution = request.form.get('ScreenResolution')
    cpu_brand = request.form.get('cpu_brand')
    hdd = int(request.form.get('hdd'))
    ssd = int(request.form.get('ssd'))
    if(ips == "No"):
        ips = 0
    if(ips == "Yes"):
        ips = 1
    if(touchscreen == "No"):
        touchscreen = 0
    if(touchscreen == "Yes"):
        touchscreen = 1  
    X_res = int(ScreenResolution.split('x')[0])
    Y_res = int(ScreenResolution.split('x')[1]) 
    ppi = ((X_res)**2 + (Y_res)**2)**0.5/size         
    input_data = [company, type_name, ram, gpu, os, weight, touchscreen, ips, ppi, cpu_brand, hdd, ssd]

    prediction = model.predict([input_data])
    prediction_antilog = np.exp(prediction)
    return render_template('index.html', prediction=prediction_antilog)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)
