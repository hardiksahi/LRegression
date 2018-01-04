# Hardik Sahi
# University of Waterloo
# Linear Regression: Implementation of Ridge algorithm

import numpy as np
import csv
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

inputfile = open('housing_X_train.csv', 'r')
inputreader = csv.reader(inputfile,delimiter=',')
inputData = list(inputreader)
inputArray = np.array(inputData).astype("float")

outputFile = open('housing_y_train.csv', 'r')
outputreader = csv.reader(outputFile, delimiter = ',')
outputData = list(outputreader)
outputArray = np.array(outputData).astype("float")
kFold = 10
lambdaWeightTrainDict = {}
kf = KFold(n_splits=kFold)

class LambdaClass():
    def __init__(self,lamda,mse):
        self.lamda = lamda
        self.mse = mse

def center_standardize_input_data(inArr):
    for i in range(inArr.shape[1]):
        maxV = np.amax(inArr[:,i])
        minV = np.amin(inArr[:,i])
        inArr[:,i] = (inArr[:,i] - minV)/(maxV-minV)
        

def meanSquaredError(inputArray, coeffVactor, outputArray,dataPointsN):
    vect = np.dot(inputArray,coeffVactor) - outputArray
    mag = np.dot(vect.transpose(),vect)
    return mag/dataPointsN
    

def ridge_regression(inputArray,outputArray,lambdaV):
    features = inputArray.shape[1]
    inputTranspose = inputArray.transpose()
    xTx = np.dot(inputTranspose,inputArray)
    identityMatrix = np.identity(features, dtype = np.float64)
    identityMatrix[features-1, features-1] = 0
    lambdaMulIdentity = identityMatrix*lambdaV 
    leftHandSideMatrix = np.add(xTx,lambdaMulIdentity)
    rightHandSideMatrix = np.dot(inputTranspose,outputArray)   
    return np.linalg.solve(leftHandSideMatrix,rightHandSideMatrix) 
    
    
lambdaListFinal = list() 
indexForFinalList = 0
#center_standardize_input_data(inputArray) 
inputArray = np.c_[inputArray,np.ones(inputArray.shape[0])] 

for lambVal in range(0,110,10):
    indexForLambdaList = 0
    lambdaErrorValList = list()
    for train_index_tuple, valid_index_tuple in kf.split(inputArray):
        copyInputArray = np.copy(inputArray)
        copyOutputArray = np.copy(outputArray)
        inputListForIndex = list(copyInputArray)
        outputListForIndex = list(copyOutputArray)
        
        inputlistTrain = list() 
        outputListTrain = list()
        inputListValid = list()
        outputListValid = list()
        
        for index in train_index_tuple:
            inputlistTrain.append(inputListForIndex[index])
            outputListTrain.append(outputListForIndex[index])
            
        for index in valid_index_tuple:
            inputListValid.append(inputListForIndex[index])
            outputListValid.append(outputListForIndex[index])
        
        trainInputArray = np.asarray(inputlistTrain)
        trainOutputArray = np.asarray(outputListTrain)
        validInputArray = np.asarray(inputListValid)
        validOutputArray = np.asarray(outputListValid)
        
        coeffVector = ridge_regression(trainInputArray,trainOutputArray,lambVal)
        dataPointsN = validInputArray.shape[0]
        mse = meanSquaredError(validInputArray,coeffVector,validOutputArray,dataPointsN)
        lambdaErrorValList.insert(indexForLambdaList,mse)
        indexForLambdaList+=1;
    meanCVErrorLamda = np.mean(lambdaErrorValList)
    obj = LambdaClass(lambVal,meanCVErrorLamda)
    lambdaListFinal.insert(indexForFinalList,obj)
    indexForFinalList+=1
    
xAxisCVPlot = []
yAxisCVPlot = []
for i in range(len(lambdaListFinal)):
    obj = lambdaListFinal[i]
    xAxisCVPlot.insert(i,obj.lamda)
    yAxisCVPlot.insert(i,obj.mse)
    print("Mean CV error for lambda %d is %f" %(obj.lamda,obj.mse))
    


# Calculating training error

xAxisTrainPlot = []
yAxisTrainPlot = []
xAxisNZPlot = []
yAxisNZPlot = []
trErrorIndex = 0

for k in range(0,110,10):
    coeffVectorTraining = ridge_regression(inputArray,outputArray,k)
    trainMSE = meanSquaredError(inputArray,coeffVectorTraining,outputArray,inputArray.shape[0])
    xAxisTrainPlot.insert(trErrorIndex,k)
    yAxisTrainPlot.insert(trErrorIndex,trainMSE[0][0])
    xAxisNZPlot.insert(trErrorIndex,k)
    yAxisNZPlot.insert(trErrorIndex,  (((coeffVectorTraining!=0).sum())/coeffVectorTraining.shape[0])*100)
    trErrorIndex+=1
    lambdaWeightTrainDict[k] = coeffVectorTraining
    print("Training error for lambda %d is %f "% (k,trainMSE))


#Read Test Data:
inputfileTest = open('housing_X_test.csv', 'r')
inputreaderTest = csv.reader(inputfileTest,delimiter=',')
inputDataTest = list(inputreaderTest)
inputArrayTest = np.array(inputDataTest).astype("float")

outputFileTest = open('housing_y_test.csv', 'r')
outputreaderTest = csv.reader(outputFileTest, delimiter = ',')
outputDataTest = list(outputreaderTest)
outputArrayTest = np.array(outputDataTest).astype("float")

inputArrayTest = np.c_[inputArrayTest,np.ones(inputArrayTest.shape[0])]

xAxisTestPlot = []
yAxisTestPlot = []
testErrorIndex = 0
for k in range(0,110,10):
    trainMSE = meanSquaredError(inputArrayTest,lambdaWeightTrainDict[k],outputArrayTest,inputArrayTest.shape[0])
    xAxisTestPlot.insert(testErrorIndex,k)
    yAxisTestPlot.insert(testErrorIndex,trainMSE[0][0])
    testErrorIndex+=1
    print("Testing error for lambda %d is %f "% (k,trainMSE))
    
#Plotting fuctions below
fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.plot(xAxisCVPlot, yAxisCVPlot, 'r-')
ax1.set_xlabel("Lambda Value")
ax1.set_ylabel("CV error")

ax2 = fig.add_subplot(222)
ax2.plot(xAxisTrainPlot, yAxisTrainPlot, 'k-')
ax2.set_xlabel("Lambda Value")
ax2.set_ylabel("Training error")

ax3 = fig.add_subplot(223)
ax3.plot(xAxisTestPlot, yAxisTestPlot, 'b-')
ax3.set_xlabel("Lambda Value")
ax3.set_ylabel("Test error")

ax4 = fig.add_subplot(224)
ax4.plot(xAxisNZPlot, yAxisNZPlot, 'b-')
ax4.set_xlabel("Lambda Value")
ax4.set_ylabel("Percentage of non zero in Weight Vector")



    

 
   

 









    