import numpy as np
import mnist
from model.network import Net
import sys
import serial 
import pygame 
import pygame.camera
from os import getenv 
from pygame.locals import * 
from datetime import datetime as dt
from preprocessing import *

#neural net training part
num_classes = 10
train_images = mnist.train_images() #[60000, 28, 28]
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
train_images -= int(np.mean(train_images))
train_images=train_images.astype('float64') 
train_images /= int(np.std(train_images))
test_images -= int(np.mean(test_images))
test_images=test_images.astype('float64') 
test_images /= int(np.std(test_images))
train_data = train_images.reshape(60000, 1, 28, 28)
train_set_labels = np.eye(num_classes)[train_labels]
testing_data = test_images.reshape(10000, 1, 28, 28)
testing_labels = np.eye(num_classes)[test_labels]

net = Net()
print('Training Lenet......')
net.train(train_data, train_set_labels, 32, 1, 'weights.pkl')
print('Testing Lenet......')
net.test(testing_data, testing_labels, 100)
print('Testing with pretrained weights......')
net.test_with_pretrained_weights(testing_data, testing_labels, 100, 'pretrained_weights.pkl')



#camera part
pygame.camera.init() 
RANGE = 300

def capture_image(): 
    cam.start()
    image = cam.get_image() 
    cam.stop()
    image=preprocess(image)
    digit, probability = net.predict_with_pretrained_weights(image, 'pretrained_weights.pkl')       #main output
    print("Number:"+str(digit)+"with Proabability:"+str(probability))
    arduino_board = serial.Serial(sys.argv[1], 9600)
    while True: 
        if arduino_board.inWaiting() > 0: 
            data = arduino_board.readline().strip()

try:
    data = int(float(data)) 
    if data <= RANGE: 
        capture_image() 
        print(data) 
except BaseException, be: 
    print(be.message)
