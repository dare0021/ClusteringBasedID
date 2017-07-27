import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as imaging

inputPath = "/home/jkih/Music/sukwoo/PAA Full Set 0725 ish/"
outputPath = inputPath + 'stats/'
pixelGraphZoom = 5

# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
PAAFeatureVectors = ['Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff']
tfColors = [(225,90,90),(20,230,40),(255,255,255)]
# TP FP FN TN Padding
compColors = [(20,230,40),(225,90,90),(240,240,60),(90,175,240),(255,255,255)]

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
		saveStats(f, accs[speaker], f1s[speaker], 'Speaker_' + speaker)
		f.write('\n')

def saveByFeature(f):
	accs = dict()
	f1s = dict()

	for result in results:
		fv = result.feature
		smartAppend(accs, fv, result.accuracy)
		smartAppend(f1s, fv, result.f1)

	accmax_bymean = [0,accs.keys()[0]]
	accmax_bymed = [0,accs.keys()[0]]
	f1max_bymean = [0,accs.keys()[0]]
	f1max_bymed = [0,accs.keys()[0]]
	f.write('Stats by Feature\n')
	for fv in accs.keys():
		saveAndPrint(f, kvpDisp('Feature', fv))
		saveStats(f, accs[fv], f1s[fv], 'Feature_' + fv)
		f.write('\n')
		accmax_bymean[0] = max(accmax_bymean[0], np.mean(accs[fv]))
		accmax_bymed[0] = max(accmax_bymed[0], np.median(accs[fv]))
		f1max_bymean[0] = max(f1max_bymean[0], np.mean(f1s[fv]))
		f1max_bymed[0] = max(f1max_bymed[0], np.median(f1s[fv]))
	f.write('\n')
	saveAndPrint(f, kvpDisp('AccMax by Mean  ', accmax_bymean))
	saveAndPrint(f, kvpDisp('AccMax by Median', accmax_bymed))
	saveAndPrint(f, kvpDisp('F1 Max by Mean  ', f1max_bymean))
	saveAndPrint(f, kvpDisp('F1 Max by Median', f1max_bymed))

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
		strlen = 7
		plt.figure()
		plt.subplot(121)
		plt.ylim([0,1])
		plt.title(plotFileNameStub + ' acc')
		plt.boxplot(accuracies)
		plt.text(.55,.95,'mean & median')
		plt.text(.55,.9, str(np.mean(accuracies))[:strlen])
		plt.text(.55,.85,str(np.median(accuracies))[:strlen])
		plt.subplot(122)
		plt.ylim([0,1])
		plt.boxplot(f1s)
		plt.text(.55,.95,'mean & median')
		plt.text(.55,.9, str(np.mean(f1s))[:strlen])
		plt.text(.55,.85,str(np.median(f1s))[:strlen])
		plt.title(plotFileNameStub + ' f1')
		assert not os.path.isfile(outputPath + plotFileNameStub + '.png')
		plt.savefig(outputPath + plotFileNameStub + '.png', bbox_inches='tight')

def saveToFile(verbose=0):
	accuracies = []
	f1s = []

	for result in results:
		accuracies.append(result.accuracy)
		f1s.append(result.f1)

	assert not os.path.isdir(outputPath)
	os.mkdir(outputPath)
	f = open(outputPath + "summary.txt", 'w')
	saveStats(f, accuracies, f1s, 'summary')

	# by speaker & feature vector
	if verbose > 0:
		saveBySpeaker(f)
		f.write("\n")
		saveByFeature(f)
		f.write("\n")

	# copy of each result file minus the raw output
	if verbose > 1:
		for result in results:
			f.write(str(result))
		f.write("\n")

	f.close()

def textToIntList(txt):
	decrement = 1
	if txt[-1] == '\n':
		decrement = 2
	txt = txt[1:len(txt)-decrement]
	txt = txt.split(', ')
	return [int(i) for i in txt]

def drawPixelGraph(numList, colorList, filePath):
	def numToColor(num):
		return colorList[num]

	width = int(len(numList)**.5)
	height = len(numList) // width
	if width * height < len(numList):
		height += 1
	assert width * height >= len(numList)
	numList = np.pad(numList, (0, width * height - len(numList)), 'constant', constant_values=len(colorList)-1)
	numList.resize(width, height)
	pxs = np.array([map(numToColor, i) for i in numList], dtype='int8')
	img = imaging.fromarray(pxs, 'RGB')
	img = img.resize((width * pixelGraphZoom, height * pixelGraphZoom))
	assert not os.path.isfile(filePath)
	img.save(filePath)

def drawPixelGraphs():
	filePaths = [inputPath + f for f in os.listdir(inputPath) if os.path.isfile(inputPath + f) and f.endswith(".log")]

	for filePath in filePaths:		
		fileName = filePath[filePath.rfind('/')+1:]
		f = open(filePath, 'r')
		f.readline()
		f.readline()
		predList = textToIntList(f.readline())
		trueList = textToIntList(f.readline())
		compList = getComparisonList(predList, trueList)
		drawPixelGraph(predList, tfColors, outputPath + fileName + '_pred.png')
		drawPixelGraph(trueList, tfColors, outputPath + fileName + '_true.png')
		drawPixelGraph(compList, compColors, outputPath + fileName + '_comp.png')
		f.close()

def getComparisonList(predList, trueList):
	retval = np.zeros(len(predList), dtype='int8')
	for i in range(len(predList)):
		if predList[i] == trueList[i]:
			if predList[i]:
				retval[i] = 0
			else:
				retval[i] = 3
		else:
			if predList[i]:
				retval[i] = 1
			else:
				retval[i] = 2
	return retval

loadFiles()
saveToFile(2)
drawPixelGraphs()