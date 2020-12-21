import argparse
import csv
import numpy as np
from random import random

# Initialize a network
def initialize_network():
	network = list()
	hidden_layer = [{'weights': [0.2,-0.3,0.4]}, {'weights': [-0.5,-0.1,-0.4]},{'weights': [0.3,0.2,0.1]}]
	network.append(hidden_layer)
	output_layer = {'weights':[-0.1,0.1,0.3,-0.4]}
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * float(inputs[i])
	return activation

# Transfer neuron activation(Sigmoid)
def transfer(activation):
	return 1.0 / (1.0 + np.exp(-activation))

# Forward propagate input to a network output
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
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * float(inputs[j])
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[int(row[-1])] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    file_path, learningRate, itr = args.data, args.eta, args.iterations    #Reading the arguments
    with open(file_path) as dataFile:
        csv_reader = csv.reader(dataFile, delimiter=',')
        dataset=list(csv_reader)
        n_inputs = len(dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in dataset]))
        network = initialize_network()
        for layer in network:
	        print(layer)
        train_network(network, dataset, 0.2, itr, n_outputs)
        
        '''for layer in network:
	        print(layer)'''

