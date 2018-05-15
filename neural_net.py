# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:29:11 2017

@author: arunabh
"""
#########################################          DATA SORTING              ##########################

import ntpath
from PIL import Image
import numpy as np
from numpy import array
from random import random
#from random import seed
from math import exp
import re
import os


#reads in an image of pgm format. 
def read_pgm(img):
    width, height = img.size
    arr = array(img)
    data = np.reshape(arr,(1,width*height))
    temp = data.astype(np.float)
    data =temp/255.0
    #print(data)
    return data
        
def get_label(path):
    ntpath.basename("/a/b/c")
    head, tail = ntpath.split(path)#angry
    if re.findall(r'left', tail):
        target_op = [0.9,0.1,0.1,0.1]#target outputs between 0-1 because sigmoid activations
        return target_op
    if re.findall(r'right', tail):#happy
        target_op = [0.1,0.9,0.1,0.1]
        return target_op
    if re.findall(r'straight', tail):#neutral
        target_op = [0.1,0.1,0.9,0.1]
        return target_op
    if re.findall(r'up', tail):#sad
        target_op = [0.1,0.1,0.1,0.9]
        return target_op

#uses the get label and read pgm functions
#returns a column vector for the full fol        
def label_vector(path):
    data = read_pgm(Image.open(path))
    label = get_label(path)
    l_data = [] 
    l_data.extend(data)
    #l_data = np.append(l_data,label)
    return l_data,label

#returns the full file list for a folder    
def file_list(path):
    items = os.listdir(path)
    newlist = []
    for names in items:
        if names.endswith(".pgm"):
            newlist.append(names)
    return(newlist) 
    
def iterate_files(mylist, tt_flag):
    inputs = []
    target_output = []
    if tt_flag==0:
        base_filename = '/home/arunabh/Python stuff/faces_4/train/'
    else:
        base_filename = '/home/arunabh/Python stuff/faces_4/test/'
    for x in mylist:
        path = os.path.join(base_filename,x)
        temp1,temp2 = label_vector(path)
        temp1 = np.transpose(temp1)
        temp2 = np.transpose(temp2)        
        #print(temp1)
        #print(temp2)
        inputs.append(temp1)
        target_output.append(temp2)
    return(inputs,target_output)

###############################################         Neural Net         ###############################################        
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network    
        
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
    
# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

#forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backprop_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors  = list()
        if i != len(network) -1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                #print(expected[j])
                errors.append(expected[j]-neuron['output'])
                #print(neuron['output'])
                #print(errors)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
            print(neuron['delta'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row
        #inputs = row[:-1]        
        #print(inputs)
        #print("end of row")        
    if i!=0:
        inputs = [neuron['output'] for neuron in network[i-1]]
    for neuron in network[i]:
        for j in range(len(inputs)):
            neuron['weights'][j] += l_rate * neuron['delta'] *inputs[j]
        neuron['weights'][-1] +=l_rate * neuron['delta']
        

    
#create filelists
trainlist = file_list('/home/arunabh/Python stuff/faces_4/train')   
testlist = file_list('/home/arunabh/Python stuff/faces_4/test')

#create final test and train vectors
train_vector,train_labels = iterate_files(trainlist,0)#tt_flag = 0 train
test_vector,test_labels = iterate_files(testlist,1)#tt_flag = 1 test

#Hyperparameters to vary
n_epoch = 10
l_rate = 0.6
n_outputs = 4
n_hidden = 6
#print "Hyperparameters "

#to initialize the same initial weights for every iteration
network = initialize_network(len(train_vector[0]), n_hidden, n_outputs)

# Train a network for a fixed number of epochs
#def train(n_epoch,train_vector,train_labels,network,)
for epoch in range(n_epoch):
    sum_error = 0
    for row,expected in zip(train_vector,train_labels):
        outputs = forward_propagate(network,row)
#        expected = [0 for i in range(n_outputs)]
#        expected[row[-1]] = 1
#        print(outputs)
#        print(expected)
        for i in range(len(expected)):
            sum_error+=(sum([expected[i]-outputs[i]])**2)/432
            
        #sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]);#sum of squared errors
        #print(sum_error)        
        backprop_error(network,expected)
        update_weights(network, row, l_rate)
    #print('>epoch=%d, error=%.3f' % (epoch, sum_error))

# Make a prediction with a network
for row,expected in zip(test_vector,test_labels):
    	prediction = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], prediction)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		prediction = new_inputs
                #print "Expected={0}, Predicted={1}".format(expected,prediction)
	#return prediction
    #prediction = forward_propagate(network,row)
    #predictions.append(prediction)
