import argparse
import csv
import numpy as np

def sigmoidNeuralNetworks(file_path, learningrate, iterations):
    data = np.genfromtxt(file_path, delimiter=',')
    #Initializing the weights
    w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o,eta= 0.2,-0.3,0.4,-0.5,-0.1,-0.4,0.3,0.2,0.1,-0.1,0.1,0.3,-0.4,learningrate;
    i=0
    #Printing the hardcoded values of first 2 lines of the output
    print('a','b','h1','h2','h3','o','t','delta_h1','delta_h2','delta_h3','delta_o','w_bias_h1','w_a_h1','w_b_h1','w_bias_h2','w_a_h2','w_b_h2','w_bias_h3','w_a_h3','w_b_h3','w_bias_o','w_h1_o','w_h2_o','w_h3_o',sep=',')
    print('-','-','-','-','-','-','-','-','-','-','-',w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o,sep=',')
    while i<iterations:
      
      for row in data:
        Input_a=row[0]
        Input_b = row[1]
        #Calculating weighted sum at the hidden layer
        net_h1 = w_a_h1 * Input_a + w_b_h1 * Input_b + w_bias_h1 * 1
        net_h2 = w_a_h2 * Input_a + w_b_h2 * Input_b + w_bias_h2 * 1
        net_h3 = w_a_h3 * Input_a + w_b_h3 * Input_b + w_bias_h3 * 1
        #Calculating the output through each neuron using sigmoid activation function
        sigmoidActivation_h1 = 1 / (1 + np.exp(-net_h1))
        sigmoidActivation_h2 = 1 / (1 + np.exp(-net_h2))
        sigmoidActivation_h3 = 1 / (1 + np.exp(-net_h3))
        net_output = sigmoidActivation_h1 * w_h1_o + sigmoidActivation_h2 * w_h2_o + sigmoidActivation_h3 * w_h3_o + 1 * w_bias_o
        sigmoidActivation_output = 1 / (1 + np.exp(-net_output))  #Net output
        error = row[2] - sigmoidActivation_output  #Error in output and actual prediction
        if error:
            #Calclating error at hidden layer
            delta_o = sigmoidActivation_output*(1-sigmoidActivation_output)*(row[2]-sigmoidActivation_output)
            delta_h1= sigmoidActivation_h1*(1-sigmoidActivation_h1)*(w_h1_o*delta_o)
            delta_h2= sigmoidActivation_h2 * (1 - sigmoidActivation_h2) * (w_h2_o  * delta_o)
            delta_h3= sigmoidActivation_h3 * (1 - sigmoidActivation_h3) * (w_h3_o * delta_o)
            #Updating the weights at hidden layer
            w_bias_h1= w_bias_h1+(eta * delta_h1 * 1)
            w_a_h1 = w_a_h1+(eta * delta_h1 * Input_a)
            w_b_h1 = w_b_h1+(eta * delta_h1 * Input_b)
            w_bias_h2 = w_bias_h2+(eta * delta_h2 * 1)
            w_a_h2 = w_a_h2+(eta * delta_h2 * Input_a)
            w_b_h2 = w_b_h2+(eta * delta_h2 * Input_b)
            w_bias_h3 = w_bias_h3+(eta * delta_h3 * 1)
            w_a_h3 = w_a_h3+(eta * delta_h3 * Input_a)
            w_b_h3 = w_b_h3+(eta * delta_h3 * Input_b)
            #Updaing the weights at output layer
            w_bias_o = w_bias_o+(eta * delta_o * 1)
            w_h1_o = w_h1_o+(eta * delta_o * sigmoidActivation_h1)
            w_h2_o = w_h2_o+(eta * delta_o * sigmoidActivation_h2)
            w_h3_o = w_h3_o+(eta * delta_o * sigmoidActivation_h3)
            arr=np.array((row[0],row[1],sigmoidActivation_h1,sigmoidActivation_h2,sigmoidActivation_h3,sigmoidActivation_output,row[2],delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o))
            arr1 = np.round(arr,5)
            print(*list(arr1),sep=',')

      i+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    file_path, learningRate, iterations = args.data, args.eta, args.iterations    #Reading the arguments
    sigmoidNeuralNetworks(file_path,learningRate,iterations)