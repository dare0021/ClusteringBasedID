import os
import numpy as np
import matplotlib.pyplot as plt

inputPath = "/home/jkih/Music/sukwoo/12:33:28.796560/"
outputPath = inputPath + 'stats/'

# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
PAAFeatureVectors = ['Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff']

results = []

class Result:
	# param order same as file name
	def __init__(self, feature, algorithm, filename, testset, acc, f1):
		self.feature = feature
		self.algorithm = algorithm
		self.filename = filename
		self.testset = testset
		self.accuracy = acc
		self.f1 = f1

	def __str__(self):
		retval  = kvpDisp("Feature  ", self.feature)
		retval += kvpDisp("Algorithm", self.algorithm)
		retval += kvpDisp("FileNum  ", self.filename)
		retval += kvpDisp("TestSet  ", self.testset)
		retval += kvpDisp("Accuracy ", self.accuracy)
		retval += kvpDisp("F1       ", self.f1)
		return retval

def kvpDisp(key, val):
	return str(key) + " : " + str(val) + "\n"

def saveAndPrint(f, s):
	print s
	f.write(s)

def getAlgoName(fileName):
	fileName = fileName[fileName.find('_')+1:fileName.rfind('_')]
	return fileName[fileName.find('_')+1:fileName.rfind('_')]

def getFeatureNum(fileName):
	fileName = fileName[fileName.find('_')+1:]
	return int(fileName[:fileName.find('_')])

def loadFiles():
	global results
	global PAAFeatureVectors

	filePaths = [inputPath + f for f in os.listdir(inputPath) if os.path.isfile(inputPath + f) and f.endswith(".log")]

	for i in range(9, 22):
		PAAFeatureVectors.append('MFCC ' + str(i))
	for i in range(22, 34):
		PAAFeatureVectors.append('Chroma Vector ' + str(i))
	PAAFeatureVectors.append('Chroma Deviation')

	for filePath in filePaths:
		f = open(filePath, 'r')
		s = f.readline()
		f.close()

		fileName = filePath[filePath.rfind('/')+1:]
		algorithm = getAlgoName(fileName)
		featureVector = "MFCC Sphinx"
		if fileName.startswith("PAA"):
			featureVector = PAAFeatureVectors[getFeatureNum(fileName)]
		accuracy = float(s[10:s.find('\t')])
		f1 = float(s[s.find('f1: ')+4:])
		fileNum = int(fileName[fileName.find(algorithm)+len(algorithm)+1:fileName.rfind('_')])
		speaker = fileName[fileName.rfind('_')+1:len(fileName)-4]
		result = Result(featureVector, algorithm, fileNum, speaker, accuracy, f1)
		print result
		results.append(result)

def smartAppend(d, key, val):
	if key in d.keys():
		d[key].append(val)
	else:
		d[key] = [val]

def saveBySpeaker(f):
	accs = dict()
	f1s = dict()

	for result in results:
		speaker = result.testset
		smartAppend(accs, speaker, result.accuracy)
		smartAppend(f1s, speaker, result.f1)

	f.write('Stats by Speaker\n')
	for speaker in accs.keys():
		saveAndPrint(f, kvpDisp('Speaker', speaker))
		saveStats(f, accs[speaker], f1s[speaker], inputPath)
		f.write('\n')

def saveByFeature(f):
	accs = dict()
	f1s = dict()

	for result in results:
		fv = result.feature
		smartAppend(accs, fv, result.accuracy)
		smartAppend(f1s, fv, result.f1)

	f.write('Stats by Feature\n')
	for fv in accs.keys():
		saveAndPrint(f, kvpDisp('Feature', fv))
		saveStats(f, accs[fv], f1s[fv])
		f.write('\n')

def saveByModel(f):
	accs = dict()
	f1s = dict()

	for result in results:
		md = result.algorithm
		smartAppend(accs, md, result.accuracy)
		smartAppend(f1s, md, result.f1)

	f.write('Stats by Model\n')
	for md in accs.keys():
		saveAndPrint(f, kvpDisp('Model', md))
		saveStats(f, accs[md], f1s[md])
		f.write('\n')

def saveStats(f, accuracies, f1s, plotFileNameStub=''):
	saveAndPrint(f, kvpDisp('Accuracy mean', np.mean(accuracies)))
	saveAndPrint(f, kvpDisp('Accuracy min ', np.min(accuracies)))
	saveAndPrint(f, kvpDisp('Accuracy max ', np.max(accuracies)))
	saveAndPrint(f, kvpDisp('Accuracy stdv', np.std(accuracies)))
	# != f1 of the whole set
	saveAndPrint(f, kvpDisp('F1 mean', np.mean(f1s)))
	saveAndPrint(f, kvpDisp('F1 min ', np.min(f1s)))
	saveAndPrint(f, kvpDisp('F1 max ', np.max(f1s)))
	saveAndPrint(f, kvpDisp('F1 stdv', np.std(f1s)))
	f.write('\n')

	if len(plotFileNameStub) > 0:
		boxplot(accuracies, plotFileNameStub + '_accuracies.png')
		boxplot(f1s, plotFileNameStub + '_f1s.png')

def saveToFile(verbose=0):
	accuracies = []
	f1s = []

	for result in results:
		accuracies.append(result.accuracy)
		f1s.append(result.f1)

	assert not os.path.isdir(outputPath)
	os.mkdir(outputPath)
	f = open(outputPath + "summary.txt", 'w')
	saveStats(f, accuracies, f1s)

	# by speaker & feature vector
	if verbose > 0:
		saveBySpeaker(f)
		f.write("\n")
		saveByFeature(f)
		f.write("\n")

	# copy of each result file minus the raw output
	if verbose > 1:
		for result in results:
			f.write(result)
		f.write("\n")

	f.close()

def boxplot(data, saveFile=''):
	plt.figure()
	plt.boxplot(data)
	# ADD %saveFile% AS TITLE
	if len(saveFile) > 0:
		plt.savefig(outputPath + saveFile)
	else:
		plt.show()

def textToBoolList(txt):
	txt = txt[1:len(txt)-1]
	txt = txt.split(', ')
	return [bool(int(i)) for i in txt]

def drawPixelGraph(f):
	f.readline()
	f.readline()
	predList = textToBoolList(f.readline())
	trueList = textToBoolList(f.readline())

loadFiles()
saveToFile(2)