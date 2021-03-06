import numpy as np
import os
import speakerInfo as sinfo
import infoSingleFile
from unpackMFC import run as unmfc
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
from datetime import datetime
import sklearn 
from threading import Thread, BoundedSemaphore
import modelStorage as mds
from enum import Enum
WindowGTVmodes = Enum('average', 'midpoint')

# primary inputs
inputPath = "/home/jkih/Music/sukwoo_2min_utt/"
manualTrainTestSet = False
trainLabels = ['kim', 'lee', 'seo', 'yoon']
testLabels = ['joo']
autoflipOutputIfBelow50 = True
# leave blank to ignore
manualTestFile = ""
manualTestDiaFilePath = "joo proc pass 3.wav.diarization.comp"
# outputPath = inputPath + '1 0.1 avg'
outputPath = inputPath + str(datetime.now().time()) + '/'
numSets = 1
numThreads = 3
printTestingTimes = True
normalizeTrainingSet = True
# if true normalizes testing set using the normalization parameters found during the training set normalization
# unless it is a single file testing set, in which case we use a per window normalization
normalizeTestSet = True
windowGTVmode = WindowGTVmodes.average

# in number of the feature vectors used. MFCC is 30ms
# large window sizes leads to OOM failure
# at least I think it's OOM; python quits silently after filling avilable RAM (16GB)
# might be able to batch SVM training? Depends on how svm.fit() works
svmWindowSize = 1000 // 30
# also in number of feature vectors
svmStride = int(svmWindowSize *.1)

# pAA settings 
# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
# in ms
windowSize = 25.625
timeStep = 10

# don't change unless necessary
zeroThresh = 1e-10
featureVectorSize = 13
threadSemaphore = BoundedSemaphore(value=numThreads)

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
			print "WARN: validateNormalization failed with mean " + str(mean)
			return False
	for stdv in featureVector.std(axis=0):
		if forgivingFloatEquivalence(stdv, 1):
			print "WARN: validateNormalization failed with stddev " + str(stdv)
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

def windowing(x, y, normalizeEachWindow = False):
	def reduceArrDimension(a):
		retval = []
		for iter in a:
			retval.extend(iter)
		return retval

	newX = []
	newY = []
	iterRange = len(x) - svmWindowSize + 1
	if iterRange % svmStride > 0:
		print "WARN: SVM window stride misaligned by:", iterRange % svmStride
	i = 0
	while i < iterRange:
		xi = x[i : i + svmWindowSize]
		if normalizeEachWindow:
			sklSS = sklearn.preprocessing.StandardScaler()
			xi = sklSS.fit_transform(xi)
		xi = reduceArrDimension(xi)
		newX.append(xi)
		if windowGTVmode == WindowGTVmodes.midpoint:
			newY.append(y[int(i + svmWindowSize / 2)])
		elif windowGTVmode == WindowGTVmodes.average:
			newY.append(round(np.mean(y[i : i + svmWindowSize])))
		else:
			print 'ERR: invalid windowGTVmode:', windowGTVmode
			assert False
		i += svmStride
	return newX, newY

# returns: feature vector array (2D), ground truth array (1D)
def collateData(speakerList, divider = None, subtractor = None, shuffle = False):
	def reduceArrDimension(a):
		retval = []
		for iter in a:
			retval.extend(iter)
		return retval
	x = []
	y = []

	for speaker in speakerList:
		if speaker in featureVectors:
			xi = featureVectors[speaker]
			yi = groundTruths[speaker]
			if shuffle:
				rng_state = np.random.get_state()
				np.random.shuffle(xi)
				np.random.set_state(rng_state)
				np.random.shuffle(yi)
		else:
			print "ERR: unknown speaker", str(speaker)
			print featureVectors.keys()
			print groundTruths.keys()
			assert False
		for i in range(len(xi)):
			x.append(xi[i])
			y.append(yi[i])

	x = reduceArrDimension(x)
	y = reduceArrDimension(y)

	sklSS = sklearn.preprocessing.StandardScaler()
	if divider == None:
		x = sklSS.fit_transform(x)		
		if not validateNormalization(x):
			print "ERR: data not normalized for speakers " + str(speakerList)
			print "Check if bounds are too close"
			assert False
	elif divider[0] == False:
		# Don't normalize
		pass
	else:
		sklSS.scale_ = divider
		sklSS.mean_ = subtractor
		x = sklSS.transform(x)
		if not validateNormalization(x):
			print "WARN: data not normalized for speakers " + str(speakerList)
			print "divider", divider
			print "subtractor", subtractor

	x, y = windowing(x, y)
	retScale = None
	retMean = None
	try:
		retScale = sklSS.scale_
		retMean = sklSS.mean_
	except AttributeError:
		pass

	return x, y, retScale, retMean

def loadManualTestFile(filePath, diarizationFilePath, divider, subtractor):
	if not (filePath in MfccCache.keys()):
		MfccCache[filePath] = unmfc(filePath, featureVectorSize)
		infoSingleFile.init(diarizationFilePath, len(MfccCache[filePath]))
	x = MfccCache[filePath]

	if not ((divider == None) or (divider[0] == False)):
		sklSS = sklearn.preprocessing.StandardScaler()
		sklSS.scale_ = divider
		sklSS.mean_ = subtractor
		x = sklSS.transform(x)

	x, y = windowing(x, infoSingleFile.getTruthValues(), True)

	x = np.array(x)
	if not validateNormalization(x):
		print "WARN: data not normalized for the manual test set"
		print "divider", divider
		print "subtractor", subtractor

	return x, y

def getSubset():
	if manualTrainTestSet:
		datA = None
		if not normalizeTrainingSet:
			datA = [False]
		trainFeatureVector, trainTruthVector, datA, datB = collateData(trainLabels, shuffle = True, divider = datA)
		if not normalizeTestSet:
			datA = [False]
		if len(manualTestFile) > 0:
			testFeatureVector, testTruthVector = loadManualTestFile(manualTestFile, manualTestDiaFilePath, datA, datB)
		else:
			testFeatureVector, testTruthVector, datA, datB = collateData(testLabels, datA, datB, True)
	else:
		global lastSpeaker
		testSpeaker = lastSpeaker + 1
		if testSpeaker >= len(featureVectors.keys()):
			testSpeaker = 0
		speakers = featureVectors.keys()
		datA = None
		if not normalizeTrainingSet:
			datA = [False]
		trainFeatureVector, trainTruthVector, datA, datB = collateData([speaker for speaker in speakers if speaker != speakers[testSpeaker]], shuffle = True, divider = datA)
		if not normalizeTestSet:
			datA = [False]
		testFeatureVector, testTruthVector, datA, datB = collateData([speakers[testSpeaker]], datA, datB, True)

		lastSpeaker = testSpeaker
		print "Testing with speaker #" + str(testSpeaker) + ", label: " + str(speakers[testSpeaker])
	return trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector

# flips 0 to 1 and non-0 to 0 for any given 1D array
def flipTruthValues(truthVect):
	def flip(item):
		if item == 0:
			return 1
		return 0
	return map(flip, truthVect)

def modelProcess(modelFunc, tag, ms):
	global threadSemaphore	
	def resetModel():
		if ms.args == 'ensembleOverride':
			return modelFunc
		elif ms.args != None:
			return modelFunc(ms.args)
		else:
			return modelFunc()
	gtvWasFlipped = False
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
		if printTestingTimes:
			print 'TESTING BEGIN', datetime.now()
		predicted_labels = model.predict(testFeatureVector)
		if printTestingTimes:
			print 'TESTING END', datetime.now()
		accuracy = model.score(testFeatureVector, testTruthVector)
		if autoflipOutputIfBelow50 and accuracy < .5:
			accuracy = 1 - accuracy
			gtvWasFlipped = True
			testTruthVector = flipTruthValues(testTruthVector)
		f1 = sklearn.metrics.f1_score(testTruthVector, predicted_labels)
	except AttributeError:
		# some models only have online modes
		if printTestingTimes:
			print 'TESTING BEGIN', datetime.now()
		predicted_labels = model.fit_predict(testFeatureVector)
		if printTestingTimes:
			print 'TESTING END', datetime.now()
		accuracy = float(pairwiseComparison(predicted_labels, testTruthVector).count(True)) / len(testTruthVector)
		if autoflipOutputIfBelow50 and accuracy < .5:
			accuracy = 1 - accuracy
			gtvWasFlipped = True
			testTruthVector = flipTruthValues(testTruthVector)
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
	f.write('\n')
	f.write('Ground Truth Values Auto-Flipped: ' + str(gtvWasFlipped))
	f.close()
	threadSemaphore.release()
	
def runPaaFunctions():
	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	for paaFunction in [21, 20]:
		print "Running feature extraction #" + str(paaFunction)
		clearVariables()
		loadFeatureVector(inputPath, 'paa', paaFunction)
		for i in range(numSets * len(featureVectors.keys())):
			trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
			ms = mds.ModelSettings(i, paaFunction, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, featureVectors.keys()[lastSpeaker])
			mds.runAllModelsPAA(ms, windowSize, iterDone, iterTotal)

def runSphinxFiles():
	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	clearVariables()
	loadFeatureVector(inputPath, 'mfcc')
	iterlen = numSets * len(featureVectors.keys())
	for i in range(iterlen):
		print "PROCESSING: " + str(i) + " / " + str(iterlen)
		trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
		ms = mds.ModelSettings(i, -1, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, featureVectors.keys()[lastSpeaker])
		mds.runAllModelsMFCC(ms, iterDone, iterTotal)

def runRBFvariants():
	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	clearVariables()
	loadFeatureVector(inputPath, 'mfcc')
	if manualTrainTestSet:
		iterlen = numSets
	else:
		iterlen = numSets * len(featureVectors.keys())
	for i in range(iterlen):
		print "PROCESSING: " + str(i) + " / " + str(iterlen)
		trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
		testSpeaker = featureVectors.keys()[lastSpeaker]
		if lastSpeaker < 0:
			testSpeaker = 'manual'
		ms = mds.ModelSettings(i, -1, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, testSpeaker)
		mds.runRBFvariantsGamma(ms, [0.015], i, iterlen)
		# mds.runRBFvariants2DList(ms, [1, 10, 50, 100], [50, 0.01, 0.02, 0.03, 0.04, 0.5, 2, .78125, .617284], i, iterlen)
		# mds.runRBFvariantsCList(ms, np.arange(1.98, 3, 0.02), 0.03, i, iterlen)
		# mds.runRBFvariantsCList(ms, [1], 0.03, i, iterlen)

def runRandomForest():
	global outputPath
	outputPathPrefix = outputPath

	clearVariables()
	loadFeatureVector(inputPath, 'mfcc')
	if manualTrainTestSet:
		iterlen = numSets
	else:
		iterlen = numSets * len(featureVectors.keys())
	forestCount = [1024, 2048, 3072, 4096, 5121, 6045, 8193]
	maxDepth = [3, 5, 10, 20]
	mds.resetETAtimer(iterlen * len(forestCount) * len(maxDepth))
	for fc in forestCount:
		for md in maxDepth:
			for i in range(iterlen):
				# outputPath = outputPathPrefix + ' ' + str(fc) + 'forests ' + str(md) + 'depth/'
				if not os.path.exists(outputPath):
					os.mkdir(outputPath)
				print "PROCESSING: " + str(i) + " / " + str(iterlen) + ' ' +  str(fc) + ' forests ' + str(md) + ' depth'
				trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
				testSpeaker = featureVectors.keys()[lastSpeaker]
				if lastSpeaker < 0:
					testSpeaker = 'manual'
				ms = mds.ModelSettings(i, -1, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, testSpeaker, mds.factory_RandomForest(fc, 4, md))
				mds.runModel(mds.model_RandomForest, 'MFCC_' + str(ms.paaFunction) + '_RandomForest_fc_' + str(fc) + '_md_' + str(md) + '_' + str(ms.i) + '_' + ms.speakerName, ms)
				mds.incrementETAtimer()

def runSvmRfEnsemble():
	clearVariables()
	loadFeatureVector(inputPath, 'mfcc')
	if manualTrainTestSet:
		iterlen = numSets
	else:
		iterlen = numSets * len(featureVectors.keys())
	forestCount = 4096
	maxDepth = 3
	gamma = 0.015
	cVal = 1
	if not os.path.exists(outputPath):
		os.mkdir(outputPath)
	mds.resetETAtimer(iterlen)
	for i in range(iterlen):
		fc = forestCount
		md = maxDepth
		g = gamma
		c = cVal		
		trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector = getSubset()
		testSpeaker = featureVectors.keys()[lastSpeaker]
		if lastSpeaker < 0:
			testSpeaker = 'manual'
		ms = mds.ModelSettings(i, -1, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, testSpeaker, 'ensembleOverride')
		mds.runModel(mds.ensemble_VotingSvmRf(g, c, fc, md), 'MFCC_' + str(ms.paaFunction) + '_E_SVMRF_fc_' + str(fc) + '_md_' + str(md) + '_' + str(ms.i) + '_' + ms.speakerName, ms)
		mds.incrementETAtimer()



mds.init(threadSemaphore, modelProcess)
# runPaaFunctions()
# runSphinxFiles()
# runRBFvariants()
# runRandomForest()
runSvmRfEnsemble()