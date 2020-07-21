from flask import Flask

import sys
import os
import glob
import re

import cv2
import matplotlib.pyplot as plt
from numpy import array
from sklearn.cluster import KMeans
import numpy

from flask import Flask, redirect, url_for, request, render_template, Response
from werkzeug.utils import secure_filename

import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

import pickle

#Response.delete_cookie()

app = Flask(__name__)

# Load Model
#model = numpy.loadtxt('models/mrf_merge.txt')

# @app.route('/<name>')
# def index(name):
#     return '<h1>Hello {}!</h1>'.format(name)
def model_predict(img_path):
    #print(img_path)
    image = cv2.imread(img_path)
    #print(image.size)
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    temp = ((len(R)-2)*(len(R[0])-2))
    avg = [None]*temp
    cov = [None]*temp
    avg1 = [None]*temp
    cov1 = [None]*temp
    avg2 = [None]*temp
    cov2 = [None]*temp
    
    #Kanal Red
    
    z = 0
    w = 0
    for x in range(1, len(R)-1):
        for y in range(1, len(R[0])-1):
            avg[z] = (float(R[x-1][y+1])+float(R[x][y+1])+float(R[x+1][y+1])+float(R[x+1][y])+float(R[x+1][y-1])+float(R[x][y-1])+float(R[x-1][y-1])+float(R[x-1][y]))/8
            z=z+1

    for x in range(1, len(R)-1):
        for y in range(1, len(R[0])-1):
            cov[w] = numpy.sqrt(((avg[w]-float(R[x-1][y+1]))**2+(avg[w]-float(R[x][y+1]))**2+(avg[w]-float(R[x+1][y+1]))**2+(avg[w]-float(R[x+1][y]))**2+(avg[w]-float(R[x+1][y-1]))**2 + (avg[w]-float(R[x][y-1]))**2 +(avg[w]-float(R[x-1][y-1]))**2 + (avg[w]-float(R[x-1][y]))**2)/8)
            w=w+1

    #Kanal Green
    z = 0
    w = 0

    for x in range(1, len(G)-1):
        for y in range(1, len(G[0])-1):
            avg1[z] = (float(G[x-1][y+1])+float(G[x][y+1])+float(G[x+1][y+1])+float(G[x+1][y])+float(G[x+1][y-1])+float(G[x][y-1])+float(G[x-1][y-1])+float(G[x-1][y]))/8
            z=z+1

    for x in range(1, len(G)-1):
        for y in range(1, len(G[0])-1):
            cov1[w] = numpy.sqrt(((avg1[w]-float(G[x-1][y+1]))**2+(avg1[w]-float(G[x][y+1]))**2+(avg1[w]-float(G[x+1][y+1]))**2+(avg1[w]-float(G[x+1][y]))**2+(avg1[w]-float(G[x+1][y-1]))**2+(avg1[w]-float(G[x][y-1]))**2+(avg1[w]-float(G[x-1][y-1]))**2+(avg1[w]-float(G[x-1][y]))**2)/8)
            w=w+1

    #Kanal Blue        
    z = 0
    w = 0

    for x in range(1, len(B)-1):
        for y in range(1, len(B[0])-1):
            avg2[z] = (float(B[x-1][y+1])+float(B[x][y+1])+float(B[x+1][y+1])+float(B[x+1][y])+float(B[x+1][y-1])+float(B[x][y-1])+float(B[x-1][y-1])+float(B[x-1][y]))/8
            z=z+1
  
    for x in range(1, len(B)-1):
        for y in range(1, len(B[0])-1):
            cov2[w] = numpy.sqrt(((avg2[w]-float(B[x-1][y+1]))**2 + (avg2[w]-float(B[x][y+1]))**2+(avg2[w]-float(B[x+1][y+1]))**2+(avg2[w]-float(B[x+1][y]))**2 +(avg2[w]-float(B[x+1][y-1]))**2+(avg2[w]-float(B[x][y-1]))**2+(avg2[w]-float(B[x-1][y-1]))**2+(avg2[w]-float(B[x-1][y]))**2)/8)
            w=w+1


    matrix2 = numpy.empty([1, temp,6])

    for y in range(0, temp):
        matrix2[0][y] = [avg[y]/2,cov[y],avg1[y]/2,cov1[y],avg2[y]/2,cov2[y]]

    X = matrix2[0]
    #print(X)
    
    pkl_filename = "models/kmeans_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    y_kmeans = pickle_model.predict(X)
    
    #kmeans = KMeans(n_clusters=5, random_state=0).fit(model)
    #y_kmeans = kmeans.predict(X)

    #Redrawing the matrix
    matrix3 = numpy.empty([len(R)-2, len(R[0])-2])
    t=0
    for x in range(0, len(matrix3)):
        for y in range(0, len(matrix3[0])):
            matrix3[x][y] = y_kmeans[t]
            t=t+1

    #Redrawing Targeted Color
    matrix4 = numpy.empty([len(R)-2, len(R[0])-2])
    t=0
    for x in range(0, len(matrix4)):
        for y in range(0, len(matrix4[0])):
            if (y_kmeans[t]==2):
                matrix4[x][y] = 1
            else:
                matrix4[x][y] = 0
            t=t+1
    
    kernel = numpy.ones((3,3),numpy.uint8)
    erosi = cv2.erode(matrix4,kernel,iterations = 1)
    
    D = ndimage.distance_transform_edt(erosi)
    localMax = peak_local_max(D, indices=False, min_distance=10,labels=erosi)
     
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=numpy.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=erosi)
    print("[INFO] {} unique segments found".format(len(numpy.unique(labels)) - 1))
    total_label = format(len(numpy.unique(labels)) - 1)
    
    plt.imsave('static/img_conv/testbc2.png', erosi, cmap='gray')
    #return render_template('index.html', name = 'new_plot', filename ='testbc2.png')
    return total_label
    #return redirect(url_for('image'))


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path)
        #preds = 'test'
        return preds
        #return render_template('index.html', name = 'new_plot', filename ='testbc2.png')
    #return render_template('index.html', name = 'new_plot', filename ='testbc2.png')
    return None

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='img_conv/' + filename), code=301)

if __name__ == "__main__":
    app.run()