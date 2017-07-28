import numpy as np
import os
import speakerInfo as sinfo
from unpackMFC import run as unmfc
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
from datetime import datetime
import sklearn 
from sklearn import neighbors, svm, cluster


# primary inputs
inputPath = "/home/jkih/Music/sukwoo/"
outputPath = inputPath + str(datetime.now().time()) + '/'
num_sets = 3

# pAA settings 
# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
# -1 for all
paaFunction = -1
# in ms
windowSize = 300
timeStep = 100

# don't change unless necessary
zeroThresh = 1e-10
featureVectorSize = 13

# no touch
featureVectors = dict()
# caching is untested and might lead to increased OOM rates
featureVectorCache = dict()
groundTruths = dict()
lastSpeaker = -1

def clearVariables():
	global featureVectors
	global groundTruths
	global lastSpeaker

	featureVectors = dict()
	groundTruths = dict()
	lastSpeaker = -1

def forgivingFloatEquivalence(value, standard):
	return value < -1 * standard - zeroThresh or value > standard + zeroThresh

def pairwiseComparison(a, b):
	retval = []
	for i, j in zip(a, b):
		if i == j:
			retval.append(True)
		else:
			retval.append(False)
	return retval

def recallCalc(test, truth):
	correct = 0
	dividend = 0
	for tst, trt in zip(test, truth):
		if trt:
			dividend += 1
			if tst:
				correct +=1
	return float(correct) / dividend

def validateNormalization(featureVector):
	for mean in featureVector.mean(axis=0):
		if forgivingFloatEquivalence(mean, 0):
			print "WARN: validationNormalization failed with mean " + str(mean)
			return False
	for stdv in featureVector.std(axis=0):
		if forgivingFloatEquivalence(stdv, 1):
			print "WARN: validationNormalization failed with stddev " + str(stdv)
			return False
	return True

def loadFeatureVector(inputPath, featureType):
	if featureType == 'mfcc':
		loadMFCCFiles(inputPath)
	elif featureType == 'paa':
		loadWAVwithPAA(inputPath)
	else:
		print "ERR: unknown feature type", featureType
		assert False

def storeFeature(sid, data, filePath):
	global featureVectors
	global groundTruths

	if sid in featureVectors:
		featureVectors[sid].append(data)
		groundTruths[sid].append(np.full(len(data), sinfo.getTruthValue(filePath), dtype='int8'))
	else:
		if type(data) is np.ndarray:
			data = data.tolist()
		featureVectors[sid] = [data]
		groundTruths[sid] = [np.full(len(data), sinfo.getTruthValue(filePath), dtype='int8').tolist()]

def loadMFCCFiles(inputPath):	
	filePaths = [inputPath+f for f in os.listdir(inputPath) if os.path.isfile(inputPath+f) and f.endswith('.mfc')]
	for filePath in filePaths:
		sid = sinfo.getSpeakerID(filePath)
		data = unmfc(filePath, featureVectorSize)
		storeFeature(sid, data, filePath)

def loadWAVwithPAA(inputPath):
	filePaths = [inputPath+f for f in os.listdir(inputPath) if os.path.isfile(inputPath+f) and f.endswith('.wav')]
	for filePath in filePaths:
		sid = sinfo.getSpeakerID(filePath)
		data = None
		if filePath in featureVectorCache.keys():
			data = featureVectorCache[filePath]
		else:
			[Fs, x] = audioBasicIO.readAudioFile(filePath)
			assert paaFunction > -1 and paaFunction < 34
			data = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.001 * windowSize * Fs, 0.001 * timeStep * Fs)
			featureVectorCache[filePath] = data
		data = data[paaFunction,:]
		# using 1D feature vector breaks my code, sklearn code, and probably the law
		if len(np.array(data).shape) < 2:
			data = [[datum] for datum in data]
		storeFeature(sid, data, filePath)

# returns: feature vector array (2D), ground truth array (1D)
def collateData(speakerList):
	x = []
	y = []

	for speaker in speakerList:
		if speaker in featureVectors:
			data = featureVectors[speaker]
		else:
			print "ERR: unknown speaker", str(speaker)
			print featureVectors.keys()
			print groundTruths.keys()
			assert False
		for i in range(len(data)):
			x.extend(data[i])
			y.extend(groundTruths[speaker][i])

	sklSS = sklearn.preprocessing.StandardScaler()
	x = sklSS.fit_transform(x)
	if not validateNormalization(x):
		print "ERR: data not normalized for speakers " + str(speakerList)
		print "Check if bounds are too close"
		assert False
	return x, y

def getSubset():
	global lastSpeaker

	testSpeaker = lastSpeaker + 1
	if testSpeaker >= len(featureVectors.keys()):
		testSpeaker = 0
	speakers = featureVectors.keys()
	testFeatureVector, testTruthVector = collateData([speakers[testSpeaker]])
	trainFeatureVector, trainTruthVector = collateData([speaker for speaker in speakers if speaker != speakers[testSpeaker]])

	lastSpeaker = testSpeaker
	print "Testing with speaker #" + str(testSpeaker) + ", label: " + str(speakers[testSpeaker])
	return trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector

def model_KNN():
	print 'Running KNN'
	return sklearn.neighbors.KNeighborsClassifier(n_neighbors=sinfo.getNbClasses())

# data set too small
def model_RNC():
	print 'Running RNC'
	return sklearn.neighbors.RadiusNeighborsClassifier()

def model_SVM_linear():
	print 'Running SVM Linear'
	return sklearn.svm.SVC(kernel='linear')

def model_SVM_poly():
	print 'Running SVM Poly'
	return sklearn.svm.SVC(kernel='poly')

# Radial Basis Function
def model_SVM_rbf():
	print 'Running SVM RBF'
	return sklearn.svm.SVC(kernel='rbf')

# not enough memory
def model_Spectral():
	print 'Running Spectral Clustering'
	return sklearn.cluster.SpectralClustering(n_clusters=sinfo.getNbClasses())

# not enough memory
def model_MiniK():
	print 'Running Mini K'
	return sklearn.cluster.MiniBatchKMeans(n_clusters=sinfo.getNbClasses())

def model_ACWard():
	print 'Running AC Ward'
	return sklearn.cluster.AgglomerativeClustering(n_clusters=sinfo.getNbClasses(), linkage='ward')

def model_ACComplete():
	print 'Running AC Complete'
	return sklearn.cluster.AgglomerativeClustering(n_clusters=sinfo.getNbClasses(), linkage='complete')

def model_ACAverage():
	print 'Running AC Average'
	return sklearn.cluster.AgglomerativeClustering(n_clusters=sinfo.getNbClasses(), linkage='average')

# clustering class incompatible with classifiers
# clustering is deterministic, non-ML? Doesn't seem to use training at all.
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
def model_Birch():
	print 'Running Birch'
	return sklearn.cluster.Birch(n_clusters=sinfo.getNbClasses())

def runModel(modelFunc, tag, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector):
	model = modelFunc()
	model.fit(trainFeatureVector, trainTruthVector)
	accuracy = -1
	f1 = -1
	try:
		if type(model) is type(model_MiniK()):
			model = modelFunc()
			model.dummyattributethatdoesntexist
			# MiniK score is not accuracy
			# raise an attribute error to skip in to the hand-written accuracy code
		predicted_labels = model.predict(testFeatureVector)
		accuracy = model.score(testFeatureVector, testTruthVector)
		f1 = sklearn.metrics.f1_score(testTruthVector, predicted_labels)
	except AttributeError:
		# some models only have online modes
		predicted_labels = model.fit_predict(testFeatureVector)
		accuracy = float(pairwiseComparison(predicted_labels, testTruthVector).count(True)) / len(testTruthVector)
		recall = recallCalc(predicted_labels, testTruthVector)
		f1 = float(2) * accuracy * recall / (accuracy + recall)
	if accuracy < 0 or accuracy > 1:
		print 'INVALID ACC ' + str(accuracy)
		print 'MODEL ' + str(model)
		print str(predicted_labels)
		print str(testTruthVector)
		os.exit
	elif f1 < 0 or f1 > 1:
		print 'INVALID F1 ' + str(f1)
		print 'MODEL ' + str(model)
		print str(predicted_labels)
		print str(testTruthVector)
		os.exit

	f = open(outputPath + tag + '.log', 'w')
	f.write('accuracy: ' + str(accuracy) + '\tf1: ' + str(f1))
	f.write('\n')
	f.write('predicted labels followed by truth values')
	f.write('\n')
	f.write(str(predicted_labels.tolist()))
	f.write('\n')
	f.write(str(testTruthVector))
	f.close()

	return accuracy, f1

def runAllModels(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector):
	runModel(model_KNN, 'PAA_' + str(paaFunction) + '_KNN_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_RNC, 'PAA_' + str(paaFunction) + '_RNC_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_linear, 'PAA_' + str(paaFunction) + '_SVM_Linear_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_poly, 'PAA_' + str(paaFunction) + '_SVM_Poly_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_rbf, 'PAA_' + str(paaFunction) + '_SVM_RBF_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_Spectral, 'PAA_' + str(paaFunction) + '_SpectralClustering_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_MiniK, 'PAA_' + str(paaFunction) + '_MiniK_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACWard, 'PAA_' + str(paaFunction) + '_ACWard_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACAverage, 'PAA_' + str(paaFunction) + '_ACAvg_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACComplete, 'PAA_' + str(paaFunction) + '_ACComplete_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_Birch, 'PAA_' + str(paaFunction) + '_Birch_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)

def runAllModelsPAA(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector):
	if windowSize >= 300:
		runModel(model_Spectral, 'PAA_' + str(paaFunction) + '_SpectralClustering_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
		runModel(model_MiniK, 'PAA_' + str(paaFunction) + '_MiniK_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)	
	runModel(model_KNN, 'PAA_' + str(paaFunction) + '_KNN_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_linear, 'PAA_' + str(paaFunction) + '_SVM_Linear_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_poly, 'PAA_' + str(paaFunction) + '_SVM_Poly_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_rbf, 'PAA_' + str(paaFunction) + '_SVM_RBF_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACWard, 'PAA_' + str(paaFunction) + '_ACWard_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACAverage, 'PAA_' + str(paaFunction) + '_ACAvg_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACComplete, 'PAA_' + str(paaFunction) + '_ACComplete_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_Birch, 'PAA_' + str(paaFunction) + '_Birch_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)

def runAllModelsMFCC(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector):
	runModel(model_KNN, 'MFCC_' + str(paaFunction) + '_KNN_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_linear, 'MFCC_' + str(paaFunction) + '_SVM_Linear_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_poly, 'MFCC_' + str(paaFunction) + '_SVM_Poly_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_rbf, 'MFCC_' + str(paaFunction) + '_SVM_RBF_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	
def runPaaFunctions():
	global paaFunction

	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	if paaFunction < 0:
		for paaFunction in range(0, 34):
			print "Running feature extraction #" + str(paaFunction)
			clearVariables()
			loadFeatureVector(inputPath, 'paa')
			for i in range(num_sets * len(featureVectors.keys())):
				trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
				runAllModelsPAA(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	else:
		clearVariables()
		loadFeatureVector(inputPath, 'paa')
		for i in range(num_sets * len(featureVectors.keys())):
			trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()		
			runAllModelsPAA(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)

def runSphinxFiles():
	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	clearVariables()
	loadFeatureVector(inputPath, 'mfcc')
	for i in range(num_sets * len(featureVectors.keys())):
		trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
		runAllModelsMFCC(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)

runPaaFunctions()
runSphinxFiles()