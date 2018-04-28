import tensorflow as tf 
import numpy as np 
import cv2
from collections import deque
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras.layers import Input,Conv2D,Dense,Flatten
#from tensorflow.python.keras.layers import Input
from keras.models import Sequential
from keras.layers import UpSampling2D,merge,MaxPooling2D
import project
import json
import random
# from keras import initializations
# from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
n_actions=3
gamma=0.99
initial_epsilon=0
final_epsilon=0
explore_steps=5000
observation_steps=100000000
replay_memory=50000
batch_size=32
learning_rate=1e-4

def preprocess(img):
	img= cv2.cvtColor(cv2.resize(img, (84, 84)), cv2.COLOR_BGR2GRAY)
	ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
	return img

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(84,84,4)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(3))
   
    adam = Adam(lr=learning_rate)
    model.compile(loss='mse',optimizer=adam)
    # print("We finish building the model")
    return model

def play(model):
	freq = np.zeros([3])
	r,img,terminal, freq=project.get_next_screen([1,0,0], freq)
	experience=deque()
	img=preprocess(img)
	s=np.stack((img,img,img,img),axis=2)
	stack=s.reshape(1,s.shape[0],s.shape[1],s.shape[2])

	t=0
	epsilon=initial_epsilon
	std_dev = 10
	action=np.zeros(n_actions)
	wins=0
	losses=0
	while(True):
		loss=0
		Q_s=0
		index=0
		reward=0

		action=np.zeros(n_actions)
		action[np.argmax(model.predict(stack))]=1
		reward,img,terminal, freq=project.get_next_screen(action, freq)

		img=preprocess(img)
		img=img.reshape(1,img.shape[0],img.shape[1],1)
		new_stack=np.append(img,stack[:,:,:,:3],axis=-1)
		if reward==-1:
			losses+=1
		if(reward ==1):
			wins+=1 
		stack=new_stack
		t+=1
        # save progress every 10000 iterations
		if t%1000==0:
			print(freq)
			print("wins",wins,"losses",losses)
			print("Now we save model")
			model.save_weights("model.h5", overwrite=True)
			with open("model.json", "w") as outfile:
				outfile.write(model.to_json())
			print("Time", t, "/ State", state1, \
		    "/ Epsilon", epsilon, "/ Action", index, "/ Reward", reward, \
		    "/ Q_max " , np.max(Q_s), "/ Loss ", loss)
        # print info
		# state = ""
		if t <= observation_steps:
		    state1 = "observe"
		elif t > observation_steps and t <= observation_steps + explore_steps:
		    state1 = "explore"
		else:
		    state1 = "train"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
json_file = open('model.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
adam = Adam(lr=learning_rate)
model.compile(loss='mse',optimizer=adam)	
# model=buildmodel()
play(model)



