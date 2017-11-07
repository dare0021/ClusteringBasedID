# processes logs in the given directory and saves them in a subdirectory "inferred" 
# after running inference model on them

import os
import ast
import numpy as np
from threading import Thread, BoundedSemaphore

# inputPath = "/media/jkih/b6988675-1154-47d9-9d37-4a80b771f7fe/new/sukwoo/ShortSegsMFCC SVM_RBF g search 0.001 0.1 0.002 non-clairvoyant/"
inputPath = "/media/jkih/b6988675-1154-47d9-9d37-4a80b771f7fe/new/sukwoo/shortsegs randomforest/1 0.1 avg fc 160 256 320 512 640 md 5 10 20 mantest/"
outputPath = inputPath + "inferred/"

# resistance to change
# e.g. 0.1 means changeover @ 0.4 and 0.6, or 0.5 +- 0.1
fuzzyness = 0.1
# decay value for EWMA
# lower decay means more inertia
decay = 1.0/2
verbose = False
saveOutputFrames = True
# file output processes
numThreads = 4

results = []
threadSemaphore = BoundedSemaphore(value=numThreads)

class Result:
	def __init__(self, filename, accuracy, f1, rawOutput, groundTruth):
		self.filename = filename
		self.accuracy = accuracy
		self.f1 = f1
		self.rawOutput = rawOutput
		self.groundTruth = groundTruth

	def __str__(self):
		retval = 'filename: ' + self.filename + '\n'
		retval += 'accuracy: ' + str(self.accuracy) + '\tf1: ' + str(self.f1)
		return retval

def loadFiles():
	global results

	filePaths = [inputPath + f for f in os.listdir(inputPath) if os.path.isfile(inputPath + f) and f.endswith(".log")]

	for filePath in filePaths:
		fileName = filePath[filePath.rfind('/')+1:]
		
		f = open(filePath, 'r')
		s = f.readline()
		accuracy = float(s[10:s.find('\t')])
		f1 = float(s[s.find('f1: ')+4:])
		s = f.readline()
		predicted = ast.literal_eval(f.readline())
		groundTruth = ast.literal_eval(f.readline())
		f.close()

		result = Result(fileName, accuracy, f1, predicted, groundTruth)
		if verbose:
			print result
		results.append(result)

def checkCrossover(weighedAverage, currentOutput):
	if currentOutput:
		return weighedAverage < 0.5 - fuzzyness
	return weighedAverage > 0.5 + fuzzyness

# async write to file
def writeToFile(path, content):
	global threadSemaphore
	def asyncWTF(path, content):		
		global threadSemaphore
		f = open(path, 'w')
		f.write(content)
		f.close()
		threadSemaphore.release()

	threadSemaphore.acquire()
	p = Thread(target=asyncWTF, args=(path, content))
	p.start()


# assumes input is 0 or 1
def processResults():
	for result in results:
		outputArray = []
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		predicted = result.rawOutput
		groundTruth = result.groundTruth
		assert len(predicted) == len(groundTruth)
		currentOutput = predicted[0] == 1
		weighedAverage = predicted[0]

		for i in range(len(predicted)):
			pi = predicted[i]
			gi = groundTruth[i] == 1
			weighedAverage = (1.0-decay) * weighedAverage + decay * pi
			if checkCrossover(weighedAverage, currentOutput):
				currentOutput = not currentOutput
			if currentOutput:
				outputArray.append(1)
				if gi:
					tp += 1
				else:
					fp += 1
			else:
				outputArray.append(0)
				if gi:
					fn += 1
				else:
					tn += 1

		accuracy = float(tp + tn) / len(predicted)
		recall = 0
		precision = 0
		f1 = 0
		if tp  != 0:
			recall = float(tp) / (tp + fn)
			precision = float(tp) / (tp + fp)
			f1 = 2.0 * precision * recall / (precision + recall)
		else:
			print "WARN: tp == 0 @ " + result.filename

		if verbose:
			print result
			print 'newAccuracy: ' + str(accuracy) + '\tnewF1: ' + str(f1)
			print 'deltaAccuracy: ' + str(accuracy - result.accuracy) + '\tdeltaF1: ' + str(f1 - result.f1)

		if not os.path.exists(outputPath):
			os.mkdir(outputPath)
		content = 'accuracy: ' + str(accuracy) + '\tf1: ' + str(f1) + '\n'
		content += 'deltaAccuracy: ' + str(accuracy - result.accuracy) + '\tdeltaF1: ' + str(f1 - result.f1) + '\n'
		if saveOutputFrames:
			content += str(outputArray) + '\n'
			content += str(result.groundTruth) + '\n'
		writeToFile(outputPath + result.filename, content)

def automatedSearch(fuzzRange, decayRange):
	global outputPath
	global fuzzyness
	global decay

	loadFiles()
	print 'File load complete'
	outputPathPrefix = outputPath[:len(outputPath)-1]
	for fi in fuzzRange:
		for di in decayRange:
			outputPath = outputPathPrefix + " f" + str(fi) + " d" + str(di) + "/"
			print outputPath
			fuzzyness = fi
			decay = di
			processResults()

loadFiles()
processResults()

# automatedSearch(np.arange(0, 0.5, 0.01), [3.0/4, 2.0/3, 1.0/2, 1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/64, 1.0/128, 1.0/256])
automatedSearch([0.0], [1.0/2])
