import numpy as np
import sklearn 
import matplotlib.pyplot as plt
import scipy, librosa, os
import speakerInfo as sinfo
import unpackMFC as unmfc
import pyAudioAnalysis as paa

# primary inputs
inputPath = "/home/jkih/Music/sukwoo/"
num_sets = 20

# pAA settings in ms
windowSize = 50
timeStep = 25

# don't change unless necessary
zeroThresh = 1e-10
featureVectorSize = 13

# no touch
featureVectors = dict()
groundTruths = dict()
lastSpeaker = -1
numSpeakers = 0

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
	elif featureType == 'centroid':
		loadCentroidsFromWAV(inputPath)
	else:
		print "ERR: unknown feature type", featureType
		assert False

def loadMFCCFiles(inputPath):	
	global featureVectors
	global groundTruths
	global numSpeakers

	numSpeakers = 0
	filePaths = [inputPath+f for f in os.listdir(inputPath) if os.path.isfile(inputPath+f) and f.endswith('.mfc')]
	for filePath in filePaths:
		sid = sinfo.getSpeakerID(filePath)
		data = unmfc.run(filePath, featureVectorSize)

		if sid in featureVectors:
			featureVectors[sid].append(data)
		else:
			featureVectors[sid] = [data.tolist()]
			groundTruths[sid] = sinfo.getTruthValue(filePath)
			numSpeakers += 1

def loadCentroidsFromWAV(inputPath):
	numSpeakers = 0
	filePaths = [inputPath+f for f in os.listdir(inputPath) if os.path.isfile(inputPath+f) and f.endswith('.wav')]
	for filePath in filePaths:
		sid = sinfo.getSpeakerID(filePath)
		[Fs, x] = paa.audioBasicIO.readAudioFile(filePath)
		data = paa.audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.001 * windowSize * Fs, 0.001 * timeStep * Fs)
		# where did the feature extraction algorithm parameter go? are they all run simultaneously? 
		# Would explain the 2D output with 34 rows (34 algorithms)

		if sid in featureVectors:
			featureVectors[sid].append(data)
		else:
			featureVectors[sid] = [data.tolist()]
			groundTruths[sid] = sinfo.getTruthValue(filePath)
			numSpeakers += 1
	
	print "ERR: feature type not implemented yet", featureType
	assert False

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
		for featureVector in data:
			x.extend(featureVector)
			y.extend(np.full((len(featureVector)), groundTruths[speaker], dtype='int8'))

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
	if testSpeaker >= numSpeakers:
		testSpeaker = 0
	speakers = featureVectors.keys()
	testFeatureVector, testTruthVector = collateData([speakers[testSpeaker]])
	trainFeatureVector, trainTruthVector = collateData([speaker for speaker in speakers if speaker != speakers[testSpeaker]])

	lastSpeaker = testSpeaker
	print "Testing with speaker #" + str(testSpeaker) + ", label: " + str(speakers[testSpeaker])
	return trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector

loadFeatureVector(inputPath, 'mfcc')
for i in range(num_sets * numSpeakers):
	getSubset()