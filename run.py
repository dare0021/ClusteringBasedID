import numpy as np
import sklearn 
import matplotlib.pyplot as plt
import scipy, librosa, os
import speakerInfo as sinfo
import unpackMFC as unmfc
import pyAudioAnalysis as paa
from datetime import datetime

# primary inputs
inputPath = "/home/jkih/Music/sukwoo/"
outputPath = inputPath + str(datetime.now().time()) + '/'
num_sets = 100

# pAA settings in ms
# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
paaFunction = 4
windowSize = 50
timeStep = 25

# don't change unless necessary
zeroThresh = 1e-10
featureVectorSize = 13

# no touch
featureVectors = dict()
groundTruths = dict()
lastSpeaker = -1

def forgivingFloatEquivalence(value, standard):
	return value < -1 * standard - zeroThresh or value > standard + zeroThresh

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
		featureVectors[sid] = [data.tolist()]
		groundTruths[sid] = [np.full(len(data), sinfo.getTruthValue(filePath), dtype='int8').tolist()]

def loadMFCCFiles(inputPath):	
	filePaths = [inputPath+f for f in os.listdir(inputPath) if os.path.isfile(inputPath+f) and f.endswith('.mfc')]
	for filePath in filePaths:
		sid = sinfo.getSpeakerID(filePath)
		data = unmfc.run(filePath, featureVectorSize)
		storeFeature(sid, data, filePath)

def loadWAVwithPAA(inputPath):
	filePaths = [inputPath+f for f in os.listdir(inputPath) if os.path.isfile(inputPath+f) and f.endswith('.wav')]
	for filePath in filePaths:
		sid = sinfo.getSpeakerID(filePath)
		[Fs, x] = paa.audioBasicIO.readAudioFile(filePath)
		data = paa.audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.001 * windowSize * Fs, 0.001 * timeStep * Fs)[paaFunction,:]
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
	return sklearn.neighbors.KNeighborsClassifier(n_neighbors=sinfo.getNbClasses())

def model_SVM():
	return sklearn.svm.SVC()

def runModel(model, tag, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector):
	model = model()
	model.fit(trainFeatureVector, trainTruthVector)
	predicted_labels = model.predict(testFeatureVector)
	accuracy = model.score(testFeatureVector, testTruthVector)
	f1 = sklearn.metrics.f1_score(testTruthVector, predicted_labels)

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



loadFeatureVector(inputPath, 'mfcc')
if not os.path.exists(outputPath):
	os.mkdir(outputPath)
for i in range(num_sets * len(featureVectors.keys())):
	trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
	runModel(model_KNN, 'KNN_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM, 'SVM_' + str(i) + '_' + featureVectors.keys()[lastSpeaker], trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)