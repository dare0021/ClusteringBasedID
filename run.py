import numpy as np
import os
import speakerInfo as sinfo
from unpackMFC import run as unmfc
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
from datetime import datetime
import sklearn 
from threading import Thread, BoundedSemaphore
import modelStorage as mds


# primary inputs
inputPath = "/home/jkih/Music/sukwoo/"
outputPath = inputPath + str(datetime.now().time()) + '/'
num_sets = 3
num_threads_sema = BoundedSemaphore(value=4)

# pAA settings 
# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
# in ms
windowSize = 25.625
timeStep = 10

# don't change unless necessary
zeroThresh = 1e-10
featureVectorSize = 13

# no touch
featureVectors = dict()
featureVectorCache = dict()
MfccCache = dict()
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

def loadFeatureVector(inputPath, featureType, paaFunction = -1):
	if featureType == 'mfcc':
		loadMFCCFiles(inputPath)
	elif featureType == 'paa':
		loadWAVwithPAA(inputPath, paaFunction)
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
		data = None
		if filePath in MfccCache.keys():
			data = MfccCache[filePath]
		else:
			data = unmfc(filePath, featureVectorSize)
			MfccCache[filePath] = data
		storeFeature(sid, data, filePath)

def loadWAVwithPAA(inputPath, paaFunction):
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

def modelProcess(modelFunc, tag, ms):
	global num_threads_sema
	def resetModel():
		if ms.args != None:
			return modelFunc(ms.args)
		else:
			return modelFunc()
	trainFeatureVector = ms.trainFeatureVector
	trainTruthVector = ms.trainTruthVector
	testFeatureVector = ms.testFeatureVector
	testTruthVector = ms.testTruthVector
	model = resetModel()
	model.fit(trainFeatureVector, trainTruthVector)
	accuracy = -1
	f1 = -1
	try:
		if modelFunc == mds.model_MiniK:
			model = resetModel()
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
	num_threads_sema.release()
	
def runPaaFunctions():
	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	for paaFunction in [21, 20]:
		print "Running feature extraction #" + str(paaFunction)
		clearVariables()
		loadFeatureVector(inputPath, 'paa', paaFunction)
		for i in range(num_sets * len(featureVectors.keys())):
			trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
			ms = mds.ModelSettings(i, paaFunction, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, featureVectors.keys()[lastSpeaker])
			mds.runAllModelsPAA(ms, windowSize)

def runSphinxFiles():
	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	clearVariables()
	loadFeatureVector(inputPath, 'mfcc')
	iterlen = num_sets * len(featureVectors.keys())
	for i in range(iterlen):
		print "PROCESSING: " + str(i) + " / " + str(iterlen)
		trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
		ms = mds.ModelSettings(i, -1, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, featureVectors.keys()[lastSpeaker])
		mds.runAllModelsMFCC(ms)

def runRBFvariants():
	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	clearVariables()
	loadFeatureVector(inputPath, 'mfcc')
	iterlen = num_sets * len(featureVectors.keys())
	for i in range(iterlen):
		print "PROCESSING: " + str(i) + " / " + str(iterlen)
		trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
		ms = mds.ModelSettings(i, -1, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, featureVectors.keys()[lastSpeaker])
		mds.runRBFvariants(ms, 0.7, 0.9, 0.01)

mds.init(num_threads_sema, modelProcess)
# runPaaFunctions()
# runSphinxFiles()
runRBFvariants()