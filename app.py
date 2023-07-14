import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))

st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand', data['Company'].unique())

# Type
type = st.selectbox('Type', data['TypeName'].unique())

# ram
ram = st.selectbox('Ram', data['Ram'].unique())

# weight
weight = st.number_input('Weight of the Laptop(in kg)')

# touchscreen
touchscreen = st.selectbox('TouchScreen', ['NO', 'YES'])

# IPS
ips = st.selectbox('IPS', ['NO', 'YES'])

# screensize
screensize = st.number_input('Screen Size(in Inches)')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768',
            '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
            '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('Cpu Brand', data['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

# ssd
ssd = st.selectbox('SSD(in GB)', [0,128,256,512,1024,2048])

# gpu
gpu = st.selectbox('GpuBrand', data['GpuBrand'].unique())

# os
os = cpu = st.selectbox('OS', data['os'].unique())

if st.button('Predict Price'):
    # form a input row, make a pipe, pass it to the model
    # query

    # calculating touchscreen
    if(touchscreen == 'Yes'):
        touchscreen = 1
    else:
        touchscreen = 0
    

    # calculating ips
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # calculating ppi
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screensize

    query = np.array([company, type, ram, weight, 
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
    # reshape it
    query = query.reshape(1, 12)
    st.title(int(np.exp(pipe.predict(query))))