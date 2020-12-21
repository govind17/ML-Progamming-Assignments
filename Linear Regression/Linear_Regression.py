
import argparse
import csv
import numpy as np

def calculatePredicatedValue(X,W_T):
    Yhat = np.dot(X,W_T)
    return Yhat

def calculateSSE(Y, Yhat):
    sse = np.sum(np.square(Yhat - Y))
    return sse

def calculateGD(W, X, Y, Yhat, lr):
    gradientDescent = X * (Y - Yhat)
    gradientDescent = np.sum(gradientDescent, axis=0)
    temp = np.array(lr * gradientDescent).reshape(W.shape)
    W = W + temp
    return gradientDescent, W

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    file_path, learningRate, threshold = args.data, args.eta, args.threshold    #Reading the arguments
    with open(file_path) as dataFile:
        reader = csv.reader(dataFile, delimiter=',')
        X = []
        Y = []
        for row in reader:
            X.append([1.0]+row[:-1])
            Y.append([row[-1]])
        n = len(X)
        X = np.array(X).astype(float)
        Y = np.array(Y).astype(float)
        W = np.zeros(X.shape[1]).astype(float)     #Initializing the weights
        W_T = W.reshape(X.shape[1], 1).round(4)    # Transpose the weight matrix
        Yhat = calculatePredicatedValue(X, W_T)    #Calculting the predicted 
        #print(yhat)
        sse_prev = calculateSSE(Y, Yhat)
        #print(sse_prev)
        print(0, *["{0:.6f}".format(val) for val in W_T.T[0]], sse_prev, sep=',')
        #Calculating Gradient Descent using X, Y, W and Predicted value with mentioned learning rate 
        gradientDescent, W_T = calculateGD(W_T, X, Y, Yhat, learningRate)
        #print(gradientDescent, W)
        i = 1
        
        while True:
            Yhat = calculatePredicatedValue(X, W_T)
            sse_new = calculateSSE(Y, Yhat)
            #print(sse_new)
            #print("inside while loop")
            #print(abs(sse_new - sse_prev))
            print(i,*["{0:.6f}".format(val) for val in W_T.T[0]], sse_new, sep=',')
            if abs(sse_new - sse_prev) > threshold:
                gradientDescent, W_T = calculateGD(W_T, X, Y, Yhat, learningRate)
                i = i + 1
                sse_prev = sse_new
            else:
                break