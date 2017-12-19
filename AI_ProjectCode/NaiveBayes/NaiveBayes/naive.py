# the code is written in python 2.7

import math
import copy 
import random
import csv


#  converting data points to floats
def strToFloats(stringEdaData):

	numOfRows = len(stringEdaData)
	for index in range(numOfRows):
		newList =[]
		for value in stringEdaData[index]:
			newList.append(float(value))
		stringEdaData[index] = newList
	return stringEdaData		

# changing all labels other than relax to one label
def sameStressLabels(observations):
	tempCopy =copy.deepcopy(observations)
	numOfRows = len(tempCopy)
	for index in range(numOfRows):
		if tempCopy[index][-1] != float(1):
			tempCopy[index][-1]=float(2)		
	return tempCopy	

# splitting dataset into 70:30 ratio for training and testing respectively
def eightyTwentySplit(observations):
	tempCopy = copy.deepcopy(observations)
	numOfRows = len(tempCopy)
	trainingData = []
	while len(trainingData) < int( 0.8* numOfRows):
		i=random.randrange(len(tempCopy))
		trainingData.append(tempCopy.pop(i))

	return [trainingData,tempCopy]	

# splitting data based on labels
def splitObsLabels(observations):
    labels =list()
    dataByLabel = dict()
    rowsInobservations = len(observations)
    for index in range(rowsInobservations):
        if observations[index][-1] not in labels:
           labels.append(observations[index][-1])
    
    for label in labels:
    	dataByLabel[label]=list()

    for index in range(rowsInobservations):
    	dataByLabel[observations[index][-1]].append(observations[index])
    return dataByLabel	

 # mean of the values
def mu(values):

    sumOfvalues = float(reduce(lambda first, rest : first+rest ,values,0))
    count = len(values)
    return sumOfvalues/count
	
 # variance of the values
def sigma(values):
	avg = mu(values)

	xsquare = sum(x*x for x in values)
	xbyN = xsquare/len(values)
	return math.sqrt(xbyN - (avg*avg))

#  mean and variance tuples
def muSigma(values):
	muSigmaValues=[]
	numOfColumns=len(values[1])
	for i in range(numOfColumns-1):
		sample=[]
		for j in range(len(values)):
			sample.append(values[j][i])
		muSigmaValues.append((mu(sample), sigma(sample)))
	return muSigmaValues		


def muSigmaOfLabels(observations):
	separated = splitObsLabels(observations)
	muSigmaValues = {}
	for key in separated.keys():
		muSigmaValues[key]= muSigma(separated[key])
	return muSigmaValues

 # gaussian probability distribution function
def gaussianProbability(feature, mu, sigma):
	e = math.exp(-(math.pow(feature-mu,2)/(math.pow(sigma,2) * 2)))
	return e * (1 /  (math.sqrt(2*math.pi) * sigma)) 
 
# probabilty of each label 
def probabiltyOfEachLabel(muSigmaValues, testDataPoint):
	probabilities = {}
	for key in muSigmaValues.keys():
		prob=0.25
		numOfColumns = len(muSigmaValues[key])
		for index in  range(numOfColumns):
			mu, sigma = muSigmaValues[key][index]
			feature = testDataPoint[index]
			prob = prob * gaussianProbability(feature,mu,sigma)
		probabilities[key] = prob	
	return 	probabilities

# predicted label based on argmax	
def predictedLabel(muSigmaValues, testDataPoint):
	dictOfProbs = probabiltyOfEachLabel(muSigmaValues, testDataPoint)
	return max(dictOfProbs,key=dictOfProbs.get)


# accuracy for 70 30 split dataset
def splitAccuray(muSigmaValues,testData):
	numOfRows = len(testData)
	count =0
	for index in range(numOfRows):
		if testData[index][-1] == predictedLabel(muSigmaValues,testData[index]):
			count += 1

	return  (float(count)/numOfRows)* 100		

# leave one out cross validation testing
def looc(observations):

    count =0
    rowsInobservations = len(observations)
    #print "Predicted      Actual"
    for i in range(rowsInobservations):
       copyOfData = copy.deepcopy(observations)
       copyOfData.pop(i)
       train = copyOfData
       test = observations[i]
       muSigmaValues = muSigmaOfLabels(train)
       result = predictedLabel(muSigmaValues, test)
       #print (str(result) + "           " + str(test[-1]))
       if result == test[-1]:
           count += 1  
    return  float(count)/rowsInobservations     

# dataset after removing data points with emotional stress label
def withoutEmotional(observations):
	tempCopy =copy.deepcopy(observations)
	tempList=[]
	rowsInobservations=len(tempCopy)
	for i in range(rowsInobservations):
		if(tempCopy[i][-1]!=float(7)):
			tempList.append(tempCopy[i])		
	return tempList		

# dataset containing datapoints only with relax and emotional stress labels
def relaxAndEmotional(observations):
	rowsInobservations=len(observations)
	aList=[]
	for i in range(rowsInobservations):
		if(observations[i][-1]==float(1) or observations[i][-1]==float(7)):
			aList.append(observations[i])	
	return aList		

# dataset containing datapoints only with relax and physical stress labels	
def relaxAndPhysical(observations):
	rowsInobservations=len(observations)
	aList=[]
	for i in range(rowsInobservations):
		if(observations[i][-1]==float(1) or observations[i][-1]==float(2)):
			aList.append(observations[i])	
	return aList		

# dataset containing datapoints only with relax and congnitive stress labels
def relaxAndCognition(observations):
	rowsInobservations=len(observations)
	newList=[]
	for i in range(rowsInobservations):
		if(observations[i][-1]==float(1) or observations[i][-1]==float(5)):
			newList.append(observations[i])	
	return newList		
	


def main():
    
    stringEdaData= list(csv.reader(open('stress.csv', "r")))
    observations = strToFloats(stringEdaData)
    acc=0
    for i in range(10000):
    	data=eightyTwentySplit(observations)
    	trainingData = data[0]
    	testintData = data[1]
    	musigmavalues = muSigmaOfLabels(trainingData)
    	accuracy = splitAccuray(musigmavalues,testintData)
    	acc=acc + accuracy;
    
    print "average accuracy for 80 20 split  over 10,000 iterations is " + str(float(acc)/10000) + "%"
   
    accuracy = looc(observations)
    print "accuracy for multiclass classification with LOOCV is " + str(accuracy*100) + "%"

 
    newObservartions = sameStressLabels(observations)
    accuracy = looc(newObservartions)
    print "accuracy for relax vs rest classification with LOOCV  is " + str(accuracy*100) + "%"
    
    newObservartions = withoutEmotional(observations)
    accuracy = looc(newObservartions)
    print "accuracy without emotional stress label  with LOOCV  is " + str(accuracy*100) + "%"

    newObservartions = relaxAndCognition(observations)
    accuracy = looc(newObservartions)
    print "accuracy for binary relax vs cognitive stress with LOOCV  is " + str(accuracy*100) + "%"

    newObservartions = relaxAndEmotional(observations)
    accuracy = looc(newObservartions)
    print " accuracy for binary relax vs emotional stress with LOOCV is " + str(accuracy*100) + "%"

    newObservartions = relaxAndPhysical(observations)
    accuracy = looc(newObservartions)
    print "accuracy for binary relax vs physical stress with LOOCV  is " + str(accuracy*100) + "%"
main()