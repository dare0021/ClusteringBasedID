import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as imaging

inputPath = "/home/jkih/Music/sukwoo/Sphinx SVM_RBF g search 0.001 0.1 0.001 non-clairvoyant/"
outputPath = inputPath + 'stats/'
pixelGraphZoom = 5

# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
PAAFeatureVectors = ['Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff']
tfColors = [(225,90,90),(20,230,40),(255,255,255)]
# TP FP FN TN Padding
compColors = [(20,230,40),(225,90,90),(240,240,60),(90,175,240),(255,255,255)]

# np.seterr(invalid='raise')
for i in range(9, 22):
	PAAFeatureVectors.append('MFCC ' + str(i))
for i in range(22, 34):
	PAAFeatureVectors.append('Chroma Vector ' + str(i))
PAAFeatureVectors.append('Chroma Deviation')

class Result:
	# param order same as file name
	def __init__(self, feature, algorithm, filename, testset, acc, f1):
		self.feature = feature
		self.algorithm = algorithm
		self.filename = filename
		self.testset = testset
		self.accuracy = acc
		self.f1 = f1
		self.type = 'Result'

	def __str__(self):
		retval  = kvpDisp("Feature  ", self.feature)
		retval += kvpDisp("Algorithm", self.algorithm)
		retval += kvpDisp("FileNum  ", self.filename)
		retval += kvpDisp("TestSet  ", self.testset)
		retval += kvpDisp("Accuracy ", self.accuracy)
		retval += kvpDisp("F1       ", self.f1)
		return retval

class CGResult:
	def __init__(self, filename, acc, f1, c, g):
		self.filename = filename
		self.accuracy = acc
		self.f1 = f1
		self.c = c
		self.gamma = g
		self.type = 'CGResult'

	def __str__(self):
		retval  = kvpDisp("FileNum  ", self.filename)
		retval += kvpDisp("Accuracy ", self.accuracy)
		retval += kvpDisp("F1       ", self.f1)
		retval += kvpDisp("c value  ", self.c)
		retval += kvpDisp("gamma    ", self.gamma)
		return retval

def kvpDisp(key, val):
	return str(key) + " : " + str(val) + "\n"

def tabSepLst(lst):
	retval = ""
	for i in lst:
		retval += str(i) + '\t'
	return retval[:len(retval)-1]

def saveAndPrint(f, s):
	print s
	f.write(s)

def getAlgoName(fileName):
	fileName = fileName[fileName.find('_')+1:fileName.rfind('_')]
	return fileName[fileName.find('_')+1:fileName.rfind('_')]

def getFeatureNum(fileName):
	fileName = fileName[fileName.find('_')+1:]
	return int(fileName[:fileName.find('_')])

def loadCGFiles(inputPath):
	results = []
	filePaths = [inputPath + f for f in os.listdir(inputPath) if os.path.isfile(inputPath + f) and f.endswith(".log")]

	for filePath in filePaths:
		f = open(filePath, 'r')
		s = f.readline()
		f.close()

		suffix = filePath[filePath.rfind('_g_')+3:]
		fileName = suffix[suffix.rfind('/')+1:]
		g = float(suffix[:suffix.find('_')])
		suffix = suffix[suffix.find('_c_')+3:]
		c = int(suffix[:suffix.find('_')])
		accuracy = float(s[10:s.find('\t')])
		f1 = float(s[s.find('f1: ')+4:])
		if accuracy < 0 and accuracy > 1:
			print 'INVALID ACC @ ' + filePath
			print accuracy
			os.exit
		elif f1 < 0 and f1 > 1:
			print 'INVALID F1 @ ' + filePath
			print f1
			os.exit
		fileNum = fileName[fileName.find('_g_')+1:fileName.rfind('_')]
		result = CGResult(fileNum, accuracy, f1, c, g)
		print result
		results.append(result)
	return results

def loadSingleVariableFiles(inputPath):
	results = []
	filePaths = [inputPath + f for f in os.listdir(inputPath) if os.path.isfile(inputPath + f) and f.endswith(".log")]

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
		if accuracy < 0 and accuracy > 1:
			print 'INVALID ACC @ ' + filePath
			print accuracy
			os.exit
		elif f1 < 0 and f1 > 1:
			print 'INVALID F1 @ ' + filePath
			print f1
			os.exit
		fileNum = int(fileName[fileName.find(algorithm)+len(algorithm)+1:fileName.rfind('_')])
		speaker = fileName[fileName.rfind('_')+1:len(fileName)-4]
		result = Result(featureVector, algorithm, fileNum, speaker, accuracy, f1)
		print result
		results.append(result)
	return results

def smartAppend(d, key, val):
	if key in d.keys():
		d[key].append(val)
	else:
		d[key] = [val]

def listFind(lst, item):
	retval = 0
	for i in lst:
		if i == item:
			return retval
		retval += 1
	return None

def saveBySpeaker(f, results):
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

def getListWithMaxFirstElement(kvp1, kvp2):
	if kvp1[0] > kvp2[0]:
		return kvp1
	return kvp2

def saveByFeature(f, results):
	accs = dict()
	f1s = dict()

	for result in results:
		fv = result.feature
		smartAppend(accs, fv, result.accuracy)
		smartAppend(f1s, fv, result.f1)

	accmax_bymean = [-1,'Null']
	accmax_bymed = [-1,'Null']
	f1max_bymean = [-1,'Null']
	f1max_bymed = [-1,'Null']
	f.write('Stats by Feature\n')
	for fv in accs.keys():
		saveAndPrint(f, kvpDisp('Feature', fv))
		saveStats(f, accs[fv], f1s[fv], 'Feature_' + fv)
		f.write('\n')
		accmax_bymean = getListWithMaxFirstElement(accmax_bymean, (np.mean(accs[fv]), fv))
		accmax_bymed = getListWithMaxFirstElement(accmax_bymed, (np.median(accs[fv]), fv))
		f1max_bymean = getListWithMaxFirstElement(f1max_bymean, (np.mean(f1s[fv]), fv))
		f1max_bymed = getListWithMaxFirstElement(f1max_bymed, (np.median(f1s[fv]), fv))
	f.write('\n')
	saveAndPrint(f, kvpDisp('AccMax by Mean  ', accmax_bymean))
	saveAndPrint(f, kvpDisp('AccMax by Median', accmax_bymed))
	saveAndPrint(f, kvpDisp('F1 Max by Mean  ', f1max_bymean))
	saveAndPrint(f, kvpDisp('F1 Max by Median', f1max_bymed))

def saveByModel(f, results):
	accs = dict()
	f1s = dict()

	for result in results:
		md = result.algorithm
		smartAppend(accs, md, result.accuracy)
		smartAppend(f1s, md, result.f1)

	accmax_bymean = [-1,'Null']
	accmax_bymed = [-1,'Null']
	f1max_bymean = [-1,'Null']
	f1max_bymed = [-1,'Null']
	f.write('Stats by Model\n')
	for md in accs.keys():
		saveAndPrint(f, kvpDisp('Model', md))
		saveStats(f, accs[md], f1s[md], 'Model_'+md)
		f.write('\n')
		accmax_bymean = getListWithMaxFirstElement(accmax_bymean, (np.mean(accs[md]), md))
		accmax_bymed = getListWithMaxFirstElement(accmax_bymed, (np.median(accs[md]), md))
		f1max_bymean = getListWithMaxFirstElement(f1max_bymean, (np.mean(f1s[md]), md))
		f1max_bymed = getListWithMaxFirstElement(f1max_bymed, (np.median(f1s[md]), md))
	saveAndPrint(f, kvpDisp('AccMax by Mean  ', accmax_bymean))
	saveAndPrint(f, kvpDisp('AccMax by Median', accmax_bymed))
	saveAndPrint(f, kvpDisp('F1 Max by Mean  ', f1max_bymean))
	saveAndPrint(f, kvpDisp('F1 Max by Median', f1max_bymed))

def saveByCombination(f, results):
	# Are nested dicts a good idea? Who knows? 
	accs = dict()
	f1s = dict()

	for result in results:
		md = result.algorithm
		fv = result.feature
		if not (md in accs.keys()):
			accs[md] = dict()
			f1s[md] = dict()
		smartAppend(accs[md], fv, result.accuracy)
		smartAppend(f1s[md], fv, result.f1)

	accmax_bymean = [-1,'Null']
	accmax_bymed = [-1,'Null']
	f1max_bymean = [-1,'Null']
	f1max_bymed = [-1,'Null']
	for md in accs.keys():
		for fv in accs[md].keys():
			saveAndPrint(f, kvpDisp('Model', md))
			saveAndPrint(f, kvpDisp('Feature', fv))
			saveStats(f, accs[md][fv], f1s[md][fv], 'Comb_'+md + ' + ' +fv)
			f.write('\n')
			accmax_bymean = getListWithMaxFirstElement(accmax_bymean, (np.mean(accs[md][fv]), md + ' + ' + fv))
			accmax_bymed = getListWithMaxFirstElement(accmax_bymed, (np.median(accs[md][fv]), md + ' + ' + fv))
			f1max_bymean = getListWithMaxFirstElement(f1max_bymean, (np.mean(f1s[md][fv]), md + ' + ' + fv))
			f1max_bymed = getListWithMaxFirstElement(f1max_bymed, (np.median(f1s[md][fv]), md + ' + ' + fv))
	saveAndPrint(f, kvpDisp('AccMax by Mean  ', accmax_bymean))
	saveAndPrint(f, kvpDisp('AccMax by Median', accmax_bymed))
	saveAndPrint(f, kvpDisp('F1 Max by Mean  ', f1max_bymean))
	saveAndPrint(f, kvpDisp('F1 Max by Median', f1max_bymed))

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
		plt.close()

def saveGrid(f, grid, cList, gList):
	saveAndPrint(f, ' \t' + tabSepLst(gList) + '\n')
	gLoc = 0
	for gSub in grid:
		saveAndPrint(f, str(cList[gLoc]) + '\t' + tabSepLst(gSub) + '\n')
		gLoc += 1

def saveCGgrid(f, results):
	accs = dict()
	f1s = dict()

	gList = []
	for result in results:
		c = result.c
		g = result.gamma
		if not (c in accs.keys()):
			accs[c] = dict()
			f1s[c] = dict()
		smartAppend(accs[c], g, result.accuracy)
		smartAppend(f1s[c], g, result.f1)
		if not (g in gList):
			gList.append(g)

	ameanGrid = np.zeros((len(accs), len(gList)))
	amedGrid = np.zeros((len(accs), len(gList)))
	fmeanGrid = np.zeros((len(accs), len(gList)))
	fmedGrid = np.zeros((len(accs), len(gList)))

	cList = accs.keys()
	cList.sort()
	gList.sort()

	accmax_bymean = [-1,'Null']
	accmax_bymed = [-1,'Null']
	f1max_bymean = [-1,'Null']
	f1max_bymed = [-1,'Null']
	for c in accs.keys():
		for g in accs[c].keys():
			saveAndPrint(f, kvpDisp('c', c))
			saveAndPrint(f, kvpDisp('g', g))
			saveStats(f, accs[c][g], f1s[c][g], 'Comb_'+str(c) + ' + ' +str(g))
			f.write('\n')
			amean = np.mean(accs[c][g])
			amed = np.median(accs[c][g])
			fmean = np.mean(f1s[c][g])
			fmed = np.median(f1s[c][g])
			cLoc = listFind(cList, c)
			gLoc = listFind(gList, g)
			ameanGrid[cLoc][gLoc] = amean
			amedGrid[cLoc][gLoc] = amed
			fmeanGrid[cLoc][gLoc] = fmean
			fmedGrid[cLoc][gLoc] = fmed
			accmax_bymean = getListWithMaxFirstElement(accmax_bymean, (amean, str(c) + ' + ' + str(g)))
			accmax_bymed = getListWithMaxFirstElement(accmax_bymed, (amed, str(c) + ' + ' + str(g)))
			f1max_bymean = getListWithMaxFirstElement(f1max_bymean, (fmean, str(c) + ' + ' + str(g)))
			f1max_bymed = getListWithMaxFirstElement(f1max_bymed, (fmed, str(c) + ' + ' + str(g)))
	saveAndPrint(f, kvpDisp('AccMax by Mean  ', accmax_bymean))
	saveAndPrint(f, kvpDisp('AccMax by Median', accmax_bymed))
	saveAndPrint(f, kvpDisp('F1 Max by Mean  ', f1max_bymean))
	saveAndPrint(f, kvpDisp('F1 Max by Median', f1max_bymed))

	csv = open(f.name + '.csv', 'w')
	saveAndPrint(csv, 'X: gamma values\n')
	saveAndPrint(csv, 'Y: c values\n')
	saveAndPrint(csv, '\n')
	saveAndPrint(csv, 'amean\n')
	saveGrid(csv, ameanGrid, cList, gList)
	saveAndPrint(csv, '\n')
	saveAndPrint(csv, 'amed\n')
	saveGrid(csv, amedGrid, cList, gList)
	saveAndPrint(csv, '\n')
	saveAndPrint(csv, 'fmean\n')
	saveGrid(csv, fmeanGrid, cList, gList)
	saveAndPrint(csv, '\n')
	saveAndPrint(csv, 'fmed\n')
	saveGrid(csv, fmedGrid, cList, gList)
	csv.close()

def saveToFile(results, verbose=0):
	accuracies = []
	f1s = []

	for result in results:
		accuracies.append(result.accuracy)
		f1s.append(result.f1)

	assert not os.path.isdir(outputPath)
	os.mkdir(outputPath)
	f = open(outputPath + "summary.txt", 'w')
	saveStats(f, accuracies, f1s, 'summary')

	if results[0].type == 'Result':
		# by broad categories
		if verbose > 0:
			saveBySpeaker(f)
			f.write("\n")
			saveByFeature(f)
			f.write("\n")
			saveByModel(f)
			f.write('\n')

		# by specific combinations of feature vector & model
		if verbose > 1:
			saveByCombination(f)
			f.write('\n')
	elif results[0].type == 'CGResult':
		if verbose > 0:
			saveCGgrid(f)
			f.write('\n')
	else:
		print "stats_script.saveToFile() failed with input:", results[0]
		assert False
		return -1

	# copy of each result file minus the raw output
	if verbose > 2:
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

def drawPixelGraphs(inputPath):
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

def variableSearchGraph(results, variableMarker, variableName, heuristicsOn = "Both"):
	accs = dict()
	f1s = dict()
	for result in results:
		modString = result.algorithm
		tfval = None
		gval = -1.0
		if modString == 'SVM_RBF_Base':
			continue
		else:
			tfval = modString[modString.rfind('_')+1:] == 'True'
			if tfval:
				tfval = 1
			else:
				tfval = 0
			gval = float(modString[modString.rfind(variableMarker)+len(variableMarker) : modString.rfind('_H_')])
		if not (gval in accs.keys()):
			accs[gval] = [[],[]]
			f1s[gval] = [[],[]]
		accs[gval][tfval].append(result.accuracy)
		f1s[gval][tfval].append(result.f1)
	keys = accs.keys()
	keys.sort()
	if heuristicsOn == 'Both' or heuristicsOn == True:
		v1 = []
		v2 = []
		v3 = []
		v4 = []
		for key in keys:
			v1.append(np.mean(accs[key][True]))
			v2.append(np.mean(f1s[key][True]))
			v3.append(np.median(accs[key][True]))
			v4.append(np.median(f1s[key][True]))
		plt.figure()
		plt.plot(keys, v1, label='Amean', color='#FF3030')
		plt.plot(keys, v2, label='Fmean', color='#FF7070')
		plt.plot(keys, v3, label='Amed', color='#3030FF')
		plt.plot(keys, v4, label='Fmed', color='#7070FF')
		plt.xticks(keys)
		plt.xlabel(variableName)
		# plt.ylim([0,1])
		plt.legend(loc=0)
		if not os.path.isdir(outputPath):
			os.mkdir(outputPath)
		assert not os.path.isfile(outputPath + variableName + '_h1.png')
		plt.savefig(outputPath + variableName + '_h1.png', bbox_inches='tight')
		plt.close()
	if heuristicsOn == 'Both' or heuristicsOn == False:
		v1 = []
		v2 = []
		v3 = []
		v4 = []
		for key in keys:
			v1.append(np.mean(accs[key][False]))
			v2.append(np.mean(f1s[key][False]))
			v3.append(np.median(accs[key][False]))
			v4.append(np.median(f1s[key][False]))
		plt.figure()
		plt.plot(keys, v1, label='Amean', color='#FF3030')
		plt.plot(keys, v2, label='Fmean', color='#FF7070')
		plt.plot(keys, v3, label='Amed', color='#3030FF')
		plt.plot(keys, v4, label='Fmed', color='#7070FF')
		plt.xticks(keys)
		plt.xlabel(variableName)
		# plt.ylim([0,1])
		plt.legend(loc=0)
		if not os.path.isdir(outputPath):
			os.mkdir(outputPath)
		assert not os.path.isfile(outputPath + variableName + '_h0.png')
		plt.savefig(outputPath + variableName + '_h0.png', bbox_inches='tight')
		plt.close()

def runMultiple():
	for di in [x[0] for x in os.walk(inputPath) if 'inferred' in x]:
		global inputPath
		global outputPath
		inputPath = "/home/jkih/Music/sukwoo/Sphinx SVM_RBF g search 0.001 0.1 0.001 non-clairvoyant/"
		outputPath = inputPath + 'stats/'
		
		results = loadSingleVariableFiles(inputPath)
		saveToFile(results, 2)
		drawPixelGraphs(inputPath)
		variableSearchGraph(results, heuristicsOn = True, variableMarker = '_g_', variableName = 'g')

# results = loadSingleVariableFilesinputPath()
# saveToFile(results, 2)
# drawPixelGraphs(inputPath)
# variableSearchGraph(results, heuristicsOn = True, variableMarker = '_g_', variableName = 'g')
runMultiple()