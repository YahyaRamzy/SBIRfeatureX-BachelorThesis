import flask
from flask import Flask,render_template,url_for,request
import pickle
import base64

import numpy as np
import cv2
import tensorflow as tf

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21;

from googleapiclient.discovery import build

def search_images(query):
    service = build("customsearch", "v1", developerKey="AIzaSyAN5U1erM17INW3tAVADPJNKOLvmZR1byo")
    res = service.cse().list(
        q=query,
        cx="d168d9d47855a4e51",
        searchType='image',
        num=10,
    ).execute()
    return res['items']

#Our dictionary
label_dict = {0:'airplane flying', 1:'apple fruit', 2:'banana fruit', 3:'bicycle', 4:'car', 5:'dog animal ',6:'door',7:'ladder',8:'crecasnt moon',9:'sheep animal',10:'table',11:'tree',12:'wheel'}


#Initializing the Default Graph (prevent errors)
graph = tf.compat.v1.get_default_graph()

# Use pickle to load in the pre-trained model.
with open(f'model_CNN1.pkl', 'rb') as f:
        modelCNN = pickle.load(f)
# Use pickle to load in the pre-trained model.
with open(f'model_cnnVGGCOMPLETE.pkl', 'rb') as f:
        modelVGG = pickle.load(f)

# Use pickle to load in the pre-trained model.
with open(f'model_resnet50commp.pkl', 'rb') as f:
        modelResnet = pickle.load(f)
        
# Use pickle to load in the pre-trained model.
with open(f'model_incep.pkl', 'rb') as f:
        modelIncep = pickle.load(f)

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/vgg16')
def vgg():
	return render_template('drawvgg.html')

@app.route('/resnet')
def resnet():
	return render_template('drawresnet.html')
@app.route('/cnn')
def cnn():
	return render_template('drawcnn.html')
@app.route('/incep')
def incep():
	return render_template('drawincep.html')
#Second route : Use our model to make prediction - render the results page.
@app.route('/predictvgg', methods=['POST', 'GET'])
def predictvgg():
        
                    final_pred = None
                    #Preprocess the image : set the image to 28x28 shape
                    #Access the image
                    draw = request.form['url']
                    #Removing the useless part of the url.
                    draw = draw[init_Base64:]
                    #Decoding
                    draw_decoded = base64.b64decode(draw)
                    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    #Resizing and reshaping to keep the ratio.
                    resized_image = cv2.resize(image, (224, 224))
                    # Convert the image to a 3-channel image
                    if len(resized_image.shape) == 2:
                                resized_image = np.stack([resized_image] * 3, axis=-1)
                    elif resized_image.shape[-1] == 1:
                                resized_image = np.concatenate([resized_image] * 3, axis=-1)

                        # Expand the dimensions of the image to add the batch dimension
                    input_image = np.expand_dims(resized_image, axis=0)
                         
                      
                    resized = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
                    vect = np.asarray(resized, dtype="uint8")
                    vect = vect.reshape(1, 1, 224, 224).astype('float32')
                    #Launch prediction
                    modelVGG.run_eagerly = True
                    my_prediction = modelVGG.predict(input_image)
                    #Getting the index of the maximum prediction
                    print(my_prediction)
                    index = np.argmax(my_prediction[0])
                    #Associating the index and its value within the dictionnary
                    final_pred = label_dict[index]
                    query = final_pred
                    results = search_images(query)
                    return render_template('results.html', results =results)


@app.route('/predictresnet', methods=['POST', 'GET'])
def predictresnet():
        
                    final_pred = None
                    #Preprocess the image : set the image to 28x28 shape
                    #Access the image
                    draw = request.form['url']
                    #Removing the useless part of the url.
                    draw = draw[init_Base64:]
                    #Decoding
                    draw_decoded = base64.b64decode(draw)
                    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    #Resizing and reshaping to keep the ratio.
                    resized_image = cv2.resize(image, (180, 180))
                    # Convert the image to a 3-channel image
                    
                    if len(resized_image.shape) == 2:
                                resized_image = np.stack([resized_image] * 3, axis=-1)
                    elif resized_image.shape[-1] == 1:
                                resized_image = np.concatenate([resized_image] * 3, axis=-1)

                        # Expand the dimensions of the image to add the batch dimension
                    input_image = np.expand_dims(resized_image, axis=0)
                         
                      
                    resized = cv2.resize(image, (180,180), interpolation = cv2.INTER_AREA)
                    vect = np.asarray(resized, dtype="uint8")
                    vect = vect.reshape(1, 1, 180, 180).astype('float32')
                    #Launch prediction
                    modelResnet.run_eagerly = True
                    my_prediction = modelResnet.predict(input_image)
                    #Getting the index of the maximum prediction
                    print(my_prediction)
                    index = np.argmax(my_prediction[0])
                    #Associating the index and its value within the dictionnary
                    final_pred = label_dict[index]
                    query = final_pred
                    results = search_images(query)
                    return render_template('results.html', results =results)

@app.route('/predictincep', methods=['POST', 'GET'])
def predictincep():
        
                    final_pred = None
                    #Preprocess the image : set the image to 28x28 shape
                    #Access the image
                    draw = request.form['url']
                    #Removing the useless part of the url.
                    draw = draw[init_Base64:]
                    #Decoding
                    draw_decoded = base64.b64decode(draw)
                    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    #Resizing and reshaping to keep the ratio.
                    resized_image = cv2.resize(image, (224, 224))
                    resized_image = resized_image / 255.0
                    # Convert the image to a 3-channel image
                    if len(resized_image.shape) == 2:
                                resized_image = np.stack([resized_image] * 3, axis=-1)
                    elif resized_image.shape[-1] == 1:
                                resized_image = np.concatenate([resized_image] * 3, axis=-1)

                        # Expand the dimensions of the image to add the batch dimension
                    input_image = np.expand_dims(resized_image, axis=0)
                         
                      
                    resized = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
                    vect = np.asarray(resized, dtype="uint8")
                    vect = vect.reshape(1, 1, 224, 224).astype('float32')
                    #Launch prediction
                    modelIncep.run_eagerly = True
                    print(input_image)
                    my_prediction = modelIncep.predict(input_image)
                    #Getting the index of the maximum prediction
                    print(my_prediction)
                    index = np.argmax(my_prediction[0])
                    #Associating the index and its value within the dictionnary
                    final_pred = label_dict[index]
                    query = final_pred
                    results = search_images(query)
                    return render_template('results.html', results =results)


@app.route('/predictcnn', methods=['POST', 'GET'])
def predictcnn():
        
                    final_pred = None
                    #Preprocess the image : set the image to 28x28 shape
                    #Access the image
                    draw = request.form['url']
                    #Removing the useless part of the url.
                    draw = draw[init_Base64:]
                    #Decoding
                    draw_decoded = base64.b64decode(draw)
                    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    #Resizing and reshaping to keep the ratio.
                    resized_image = cv2.resize(image, (224, 224))
                    resized_image = resized_image / 255.0
                    # Convert the image to a 3-channel image
                    if len(resized_image.shape) == 2:
                                resized_image = np.stack([resized_image] * 3, axis=-1)
                    elif resized_image.shape[-1] == 1:
                                resized_image = np.concatenate([resized_image] * 3, axis=-1)

                        # Expand the dimensions of the image to add the batch dimension
                    input_image = np.expand_dims(resized_image, axis=0)
                         
                      
                    resized = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
                    vect = np.asarray(resized, dtype="uint8")
                    vect = vect.reshape(1, 1, 224, 224).astype('float32')
                    #Launch prediction
                    modelCNN.run_eagerly = True
                    print(input_image)
                    my_prediction = modelCNN.predict(input_image)
                    #Getting the index of the maximum prediction
                    print(my_prediction)
                    index = np.argmax(my_prediction[0])
                    #Associating the index and its value within the dictionnary
                    final_pred = label_dict[index]
                    query = final_pred
                    results = search_images(query)
                    return render_template('results.html', results =results)




# if we deploy on cloud like aws 
#if __name__ == '__main__':
#	app.run(host = '0.0.0.0' , port = 8080)
	
    
	

# if we deploy it on our local server then use this
if __name__ == '__main__':
	app.run(debug=True)





