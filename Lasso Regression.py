# Hardik Sahi
# University of Waterloo
# Linear Regression: Implementation of Lasso algorithm

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
        
def meanSquaredError(inputArray, coeffVactor, outputArray,dataPointsN):
    vect = (np.dot(inputArray,coeffVactor)).reshape(inputArray.shape[0],1) - outputArray
    mag = np.dot(vect.transpose(),vect)
    return mag/dataPointsN

def preProcessColumns(wtVector,inputArray,outputArray):
    preProcessCol = np.zeros(inputArray.shape[0])
    for k in range(inputArray.shape[1]):
        preProcessCol = np.add(preProcessCol,inputArray[:,k]*wtVector[k])
    #print("Shape of preProcessCol:", preProcessCol.shape)
    return preProcessCol
        

def softThresholding(lambdaV, wtVal):
    if abs(wtVal)>lambdaV:
        if wtVal>0:
            return wtVal-lambdaV
        elif wtVal<0:
            return wtVal+lambdaV
    else:
        return 0
    
def lassoRegression(inputArray,outputArray,lambdaV):
    predictorN = inputArray.shape[1];
    dataPointsN = inputArray.shape[0]
    weightVector =np.zeros(predictorN)
    oldWtVector = np.zeros(predictorN)
    preProcessArray = preProcessColumns(weightVector, inputArray, outputArray)
    tolerance = 10**-3
    itCount = 0
    
    while(True):
        itCount+=1
        oldWtVector = np.copy(weightVector)
        for column in range(0,predictorN): ## iteration over all predictors
            currentCol = inputArray[:,column] #(306,)
            wtColumn = weightVector[column]
            preProcessArray = preProcessArray - currentCol*wtColumn
            C = preProcessArray.reshape(dataPointsN,1)
            C1 = np.subtract(C,outputArray)
            num = -np.dot(C1.transpose(),currentCol)
            den = np.dot(currentCol.transpose(),currentCol)
            C2 = num/den
            lambdaSt = lambdaV/den
            updatedWtForCol = softThresholding(lambdaSt,C2)
            weightVector[column] = updatedWtForCol
            preProcessArray = np.add(preProcessArray,currentCol*updatedWtForCol)
        if abs(np.dot(weightVector.transpose(),weightVector) - np.dot(oldWtVector.transpose(),oldWtVector)) < tolerance:
            break;
    return weightVector


lambdaListFinal = list() 
indexForFinalList = 0

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
        
        coeffVector = lassoRegression(trainInputArray,trainOutputArray,lambVal)
        dataPointsN = validInputArray.shape[0]
        mse = meanSquaredError(validInputArray,coeffVector,validOutputArray,dataPointsN)
        #print("MSE", mse.shape)
        lambdaErrorValList.insert(indexForLambdaList,mse)
        indexForLambdaList+=1;   
    #print(lambdaErrorValList)
    meanCVErrorLamda = np.mean(lambdaErrorValList)
    obj = LambdaClass(lambVal,meanCVErrorLamda)
    lambdaListFinal.insert(indexForFinalList,obj)
    indexForFinalList+=1
    
xAxisCVPlot = []
yAxisCVPlot = []
for i in range(len(lambdaListFinal)):
    obj = lambdaListFinal[i]
    #plt.plot()
    xAxisCVPlot.insert(i,obj.lamda)
    yAxisCVPlot.insert(i,obj.mse)
    print("Mean CV error for lambda %d is %f" %(obj.lamda, obj.mse))
    

# Calculating training error

xAxisTrainPlot = []
yAxisTrainPlot = []
xAxisNZPlot = []
yAxisNZPlot = []
trErrorIndex = 0

for k in range(0,110,10):
    coeffVectorTraining = lassoRegression(inputArray,outputArray,k)
    trainMSE = meanSquaredError(inputArray,coeffVectorTraining,outputArray,inputArray.shape[0])
    xAxisTrainPlot.insert(trErrorIndex,k)
    yAxisTrainPlot.insert(trErrorIndex,trainMSE[0][0])
    
    xAxisNZPlot.insert(trErrorIndex,k)
    yAxisNZPlot.insert(trErrorIndex,  (((coeffVectorTraining!=0).sum())/(coeffVectorTraining).shape[0])*100)
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

ax3 = fig.add_subplot(224)
ax3.plot(xAxisNZPlot, yAxisNZPlot, 'b-')
ax3.set_xlabel("Lambda Value")
ax3.set_ylabel("Percentage of non zero in Weight Vector")
