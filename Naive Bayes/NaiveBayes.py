import argparse
import numpy as np
import csv
import sys

def NaiveBayes(file_path):
    # Importing the dataset
    data = np.genfromtxt(file_path, delimiter=",",dtype='unicode')
    no_rows=np.shape(data)[0]
    no_columns=np.shape(data)[1]
    countA=0;countB=0;mean1A=0;mean2A=0;mean1B=0;mean2B=0;sum1A=0;sum2A=0;sum1B=0;sum2B=0
    for row in data:
        if(row[0]=='A'):countA=countA+1
        if(row[0]=='B'):countB=countB+1
    probabilityA=countA/(countA+countB)
    probabilityB=countB/(countA+countB)
    for row in data:
        if(row[0]=='A'):
            sum1A=sum1A+float(row[1])
            sum2A=sum2A+float(row[2])
            mean1A=round(sum1A/countA,6)
            mean2A=round(sum2A/countA,6)
        if(row[0]=='B'):
            sum1B=sum1B+float(row[1])
            sum2B=sum2B+float(row[2])
            mean1B=round(sum1B/countA,6)
            mean2B=round(sum2B/countA,6)
    sum1A=0;sum2A=0;sum1B=0;sum2B=0
    for row in data:
        if(row[0]=='A'):
            sum1A=sum1A+[(float(row[1])-mean1A)*(float(row[1])-mean1A)][0]
            variance1A=round(sum1A/(countA-1),6)
            sum2A=sum2A+[(float(row[2])-mean2A)*(float(row[2])-mean2A)][0]
            variance2A=round(sum2A/(countA-1),6)
        if(row[0]=='B'):
            sum1B=sum1B+[(float(row[1])-mean1B)*(float(row[1])-mean1B)][0]
            variance1B=round(sum1B/(countB-1),6)
            sum2B=sum2B+[(float(row[2])-mean2B)*(float(row[2])-mean2B)][0]
            variance2B=round(sum2B/(countB-1),6)
    count=0
    for row in data:
        px1A=round((1/(np.sqrt(2*np.pi*variance1A)))*(np.exp(-1*((np.square(float(row[1])-mean1A))/(2*variance1A)))),5)
        px1B=round((1/(np.sqrt(2*np.pi*variance1B)))*(np.exp(-1*((np.square(float(row[1])-mean1B))/(2*variance1B)))),5)
        px2A=round((1/(np.sqrt(2*np.pi*variance2A)))*(np.exp(-1*((np.square(float(row[2])-mean2A))/(2*variance2A)))),5)
        px2B=round((1/(np.sqrt(2*np.pi*variance2B)))*(np.exp(-1*((np.square(float(row[2])-mean2B))/(2*variance2B)))),5)
        pAX=(probabilityA*px1A*px2A)/((probabilityA*px1A*px2A)+(probabilityB*px1B*px2B))
        pBX=(probabilityB*px1B*px2B)/((probabilityA*px1A*px2A)+(probabilityB*px1B*px2B))
        if(pAX>pBX): classification='A'
        if(pAX<pBX): classification='B'
        if(row[0]!=classification):count=count+1
    print(mean1A,variance1A,mean2A,variance2A,probabilityA,sep=" ")
    print(mean1B,variance1B,mean2B,variance2B,probabilityB,sep=" ")
    print(count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str)
    args = parser.parse_args()
    file_path = args.data               #Reading the arguments
    NaiveBayes(file_path)