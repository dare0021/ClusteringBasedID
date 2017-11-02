import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as imaging
from multiprocessing import Process, BoundedSemaphore
import time

silence = True
pixelGraphZoom = 5
numThreads = 4
# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
PAAFeatureVectors = ['Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff']
tfColors = [(225,90,90),(20,230,40),(255,255,255)]
# TP FP FN TN Padding
compColors = [(20,230,40),(225,90,90),(240,240,60),(90,175,240),(255,255,255)]

np.seterr(invalid='raise')
for i in range(9, 22):
	PAAFeatureVectors.append('MFCC ' + str(i))
for i in range(22, 34):
	PAAFeatureVectors.append('Chroma Vector ' + str(i))
PAAFeatureVectors.append('Chroma Deviation')
threadSemaphore = BoundedSemaphore(value=numThreads)

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

class Result2D:
	def __init__(self, filename, acc, f1, aName, bName, aVal, bVal):
		self.filename = filename
		self.accuracy = acc
		self.f1 = f1
		self.aVal = aVal
		self.bVal = bVal
		self.type = 'Result2D'
		self.aName = aName
		self.bName = bName

	def __str__(self):
		retval  = kvpDisp("FileNum  ", self.filename)
		retval += kvpDisp("Accuracy ", self.accuracy)
		retval += kvpDisp("F1       ", self.f1)
		retval += kvpDisp(aName + "\t", self.aVal)
		retval += kvpDisp(bName + "\t", self.bVal)
		return retval

def kvpDisp(key, val):
	return str(key) + " : " + str(val) + "\n"

def tabSepLst(lst):
	retval = ""
	for i in lst:
		retval += str(i) + '\t'
	return retval[:len(retval)-1]

def saveAndPrint(f, s):
	if not silence:
		print s
	f.write(s)

def getAlgoName(fileName):
	fileName = fileName[fileName.find('_')+1:fileName.rfind('_')]
	return fileName[fileName.find('_')+1:fileName.rfind('_')]

def getFeatureNum(fileName):
	fileName = fileName[fileName.find('_')+1:]
	return int(fileName[:fileName.find('_')])

def loadTwoVariableFiles(inputPath, aKey, bKey, aName, bName):
	results = []
	filePaths = [inputPath + f for f in os.listdir(inputPath) if os.path.isfile(inputPath + f) and f.endswith(".log")]

	for filePath in filePaths:
		f = open(filePath, 'r')
		s = f.readline()
		f.close()

		suffix = filePath[filePath.rfind(aKey)+len(aKey):]
		fileName = suffix[suffix.rfind('/')+1:]
		aVal = float(suffix[:suffix.find('_')])
		suffix = suffix[suffix.find(bKey)+len(bKey):]
		bVal = int(suffix[:suffix.find('_')])
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
		fileNum = fileName[fileName.find(aKey)+1:fileName.rfind('_')]
		result = Result2D(fileNum, accuracy, f1, aName, bName, aVal, bVal)
		if not silence:
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
		if not silence:
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

def saveBySpeaker(f, results, outputPath):
	accs = dict()
	f1s = dict()

	for result in results:
		speaker = result.testset
		smartAppend(accs, speaker, result.accuracy)
		smartAppend(f1s, speaker, result.f1)

	f.write('Stats by Speaker\n')
	for speaker in accs.keys():
		saveAndPrint(f, kvpDisp('Speaker', speaker))
		saveStats(f, accs[speaker], f1s[speaker], outputPath, 'Speaker_' + speaker)
		f.write('\n')

def getListWithMaxFirstElement(kvp1, kvp2):
	if kvp1[0] > kvp2[0]:
		return kvp1
	return kvp2

def saveByFeature(f, results, outputPath):
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
		saveStats(f, accs[fv], f1s[fv], outputPath, 'Feature_' + fv)
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

def saveByModel(f, results, outputPath):
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
		saveStats(f, accs[md], f1s[md], outputPath, 'Model_'+md)
		f.write('\n')
		accmax_bymean = getListWithMaxFirstElement(accmax_bymean, (np.mean(accs[md]), md))
		accmax_bymed = getListWithMaxFirstElement(accmax_bymed, (np.median(accs[md]), md))
		f1max_bymean = getListWithMaxFirstElement(f1max_bymean, (np.mean(f1s[md]), md))
		f1max_bymed = getListWithMaxFirstElement(f1max_bymed, (np.median(f1s[md]), md))
	saveAndPrint(f, kvpDisp('AccMax by Mean  ', accmax_bymean))
	saveAndPrint(f, kvpDisp('AccMax by Median', accmax_bymed))
	saveAndPrint(f, kvpDisp('F1 Max by Mean  ', f1max_bymean))
	saveAndPrint(f, kvpDisp('F1 Max by Median', f1max_bymed))

def saveByCombination(f, results, outputPath):
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
			saveStats(f, accs[md][fv], f1s[md][fv], outputPath, 'Comb_'+md + ' + ' +fv)
			f.write('\n')
			accmax_bymean = getListWithMaxFirstElement(accmax_bymean, (np.mean(accs[md][fv]), md + ' + ' + fv))
			accmax_bymed = getListWithMaxFirstElement(accmax_bymed, (np.median(accs[md][fv]), md + ' + ' + fv))
			f1max_bymean = getListWithMaxFirstElement(f1max_bymean, (np.mean(f1s[md][fv]), md + ' + ' + fv))
			f1max_bymed = getListWithMaxFirstElement(f1max_bymed, (np.median(f1s[md][fv]), md + ' + ' + fv))
	saveAndPrint(f, kvpDisp('AccMax by Mean  ', accmax_bymean))
	saveAndPrint(f, kvpDisp('AccMax by Median', accmax_bymed))
	saveAndPrint(f, kvpDisp('F1 Max by Mean  ', f1max_bymean))
	saveAndPrint(f, kvpDisp('F1 Max by Median', f1max_bymed))

def saveStats(f, accuracies, f1s, outputPath, plotFileNameStub=''):
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

def save2Dgrid(f, results, outputPath):
	accs = dict()
	f1s = dict()

	gList = []
	aName = results[0].aName
	bName = results[0].bName
	for result in results:
		aVal = result.aVal
		bVal = result.bVal
		if not (aVal in accs.keys()):
			accs[aVal] = dict()
			f1s[aVal] = dict()
		smartAppend(accs[aVal], bVal, result.accuracy)
		smartAppend(f1s[aVal], bVal, result.f1)
		if not (bVal in gList):
			gList.append(bVal)

	ameanGrid = np.zeros((len(accs), len(gList)))
	amedGrid = np.zeros((len(accs), len(gList)))
	fmeanGrid = np.zeros((len(accs), len(gList)))
	fmedGrid = np.zeros((len(accs), len(gList)))

	aList = accs.keys()
	aList.sort()
	bList.sort()

	accmax_bymean = [-1,'Null']
	accmax_bymed = [-1,'Null']
	f1max_bymean = [-1,'Null']
	f1max_bymed = [-1,'Null']
	for aVal in accs.keys():
		for bVal in accs[aVal].keys():
			saveAndPrint(f, kvpDisp(aName, aVal))
			saveAndPrint(f, kvpDisp(bName, bVal))
			saveStats(f, accs[aVal][bVal], f1s[aVal][bVal], outputPath, 'Comb_'+str(aVal) + ' + ' +str(bVal))
			f.write('\n')
			amean = np.mean(accs[aVal][bVal])
			amed = np.median(accs[aVal][bVal])
			fmean = np.mean(f1s[aVal][bVal])
			fmed = np.median(f1s[aVal][bVal])
			aLoc = listFind(aList, aVal)
			bLoc = listFind(bList, bVal)
			ameanGrid[aLoc][bLoc] = amean
			amedGrid[aLoc][bLoc] = amed
			fmeanGrid[aLoc][bLoc] = fmean
			fmedGrid[aLoc][bLoc] = fmed
			accmax_bymean = getListWithMaxFirstElement(accmax_bymean, (amean, str(aVal) + ' + ' + str(bVal)))
			accmax_bymed = getListWithMaxFirstElement(accmax_bymed, (amed, str(aVal) + ' + ' + str(bVal)))
			f1max_bymean = getListWithMaxFirstElement(f1max_bymean, (fmean, str(aVal) + ' + ' + str(bVal)))
			f1max_bymed = getListWithMaxFirstElement(f1max_bymed, (fmed, str(aVal) + ' + ' + str(bVal)))
	saveAndPrint(f, kvpDisp('AccMax by Mean  ', accmax_bymean))
	saveAndPrint(f, kvpDisp('AccMax by Median', accmax_bymed))
	saveAndPrint(f, kvpDisp('F1 Max by Mean  ', f1max_bymean))
	saveAndPrint(f, kvpDisp('F1 Max by Median', f1max_bymed))

	csv = open(f.name + '.csv', 'w')
	saveAndPrint(csv, 'Y: ' + aName + ' values\n')
	saveAndPrint(csv, 'X: ' + bName + ' values\n')
	saveAndPrint(csv, '\n')
	saveAndPrint(csv, 'amean\n')
	saveGrid(csv, ameanGrid, aList, bList)
	saveAndPrint(csv, '\n')
	saveAndPrint(csv, 'amed\n')
	saveGrid(csv, amedGrid, aList, bList)
	saveAndPrint(csv, '\n')
	saveAndPrint(csv, 'fmean\n')
	saveGrid(csv, fmeanGrid, aList, bList)
	saveAndPrint(csv, '\n')
	saveAndPrint(csv, 'fmed\n')
	saveGrid(csv, fmedGrid, aList, bList)
	csv.close()

def saveToFile(results, outputPath, verbose=0):
	accuracies = []
	f1s = []

	for result in results:
		accuracies.append(result.accuracy)
		f1s.append(result.f1)

	assert not os.path.isdir(outputPath)

	os.mkdir(outputPath)
	f = open(outputPath + "summary.txt", 'w')
	saveStats(f, accuracies, f1s, outputPath, 'summary')

	if results[0].type == 'Result':
		# by broad categories
		if verbose > 0:
			saveBySpeaker(f, results, outputPath)
			f.write("\n")
			saveByFeature(f, results, outputPath)
			f.write("\n")
			saveByModel(f, results, outputPath)
			f.write('\n')

		# by specific combinations of feature vector & model
		if verbose > 1:
			saveByCombination(f, results, outputPath)
			f.write('\n')
	elif results[0].type == 'Result2D':
		if verbose > 0:
			save2Dgrid(f, results, outputPath)
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
	return [int(float(i)) for i in txt]

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

def drawPixelGraphs(inputPath, outputPath):
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

def variableSearchGraph(results, variableMarker, variableName, outputPath, terminatorMarker = "_"):
	accs = dict()
	f1s = dict()
	for result in results:
		modString = result.algorithm
		gval = -1.0
		if 'Base' in modString:
			continue
		else:
			gval = float(modString[modString.rfind(variableMarker)+len(variableMarker) : modString.rfind(terminatorMarker)])
		if not (gval in accs.keys()):
			accs[gval] = []
			f1s[gval] = []
		accs[gval].append(result.accuracy)
		f1s[gval].append(result.f1)
	keys = accs.keys()
	keys.sort()

	v1 = []
	v2 = []
	v3 = []
	v4 = []
	for key in keys:
		v1.append(np.mean(accs[key]))
		v2.append(np.mean(f1s[key]))
		v3.append(np.median(accs[key]))
		v4.append(np.median(f1s[key]))
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
	assert not os.path.isfile(outputPath + variableName + '.png')
	plt.savefig(outputPath + variableName + '.png', bbox_inches='tight')
	plt.close()

def asyncOp(inputPath, outputPath):
	if os.path.isdir(outputPath):
		print "WARN: output path" + outputPath + "already exists; skipping"
		threadSemaphore.release()
		return
	results = loadSingleVariableFiles(inputPath)
	saveToFile(results, outputPath, 2)
	drawPixelGraphs(inputPath, outputPath)
	# variableSearchGraph(results, variableMarker = '_g_', variableName = 'g', outputPath = outputPath)
	variableSearchGraph(results, variableMarker = '_fc_', variableName = 'forestCount', outputPath = outputPath)
	threadSemaphore.release()

# causes error on exit
# ../src/unix/threadpsx.cpp(1787): assert "wxThread::IsMain()" failed in OnExit(): only main thread can be here [in thread 7f6188876700]
# ../src/common/socket.cpp(767): assert "wxIsMainThread()" failed in IsInitialized(): unsafe to call from other threads [in thread 7f6188876700]
# likely due to the main thread exiting before the child threads
# there's (probably) no problems in the correctness of the output
def runMultiple(parentDir):
	for di in [x[0] for x in os.walk(parentDir) if ('inferred' in x[0] and not ('stats' in x[0]))]:
		threadSemaphore.acquire()
		print 'Launching async instance for:', di
		p = Process(target=asyncOp, args=(di + '/', di + '/stats/'))
		p.start()

inputPath = "/media/jkih/b6988675-1154-47d9-9d37-4a80b771f7fe/new/sukwoo/shortsegs randomforest/1 0.1 avg/"
outputPath = inputPath + 'stats/'
results = loadSingleVariableFiles(inputPath)
saveToFile(results, outputPath, 2)
drawPixelGraphs(inputPath, outputPath)
variableSearchGraph(results, variableMarker = '_fc_', variableName = 'forestCount', outputPath = outputPath, terminatorMarker = '_md')
# runMultiple(inputPath)
# runMultiple("/home/jkih/Music/sukwoo_2min_utt/5s window 0.3333 stride/")