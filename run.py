import numpy as np
import sklearn as ml
import matplotlib.pyplot as plt
import scipy, librosa, os
import speakerInfo as sinfo
import unpackMFC as unmfc

inputPath = "/home/jkih/Music/sukwoo/"
featureVectorSize = 13

featureVectors = dict()
groundTruths = dict()

def LoadMFCCFiles(inputPath):	
	filePaths = [inputPath+f for f in os.listdir(inputPath) if os.path.isfile(inputPath+f) and f.endswith('.mfc')]
	for filePath in filePaths:
		sid = sinfo.getSpeakerID(filePath)
		data = unmfc.run(filePath, featureVectorSize)
		if sid in featureVectors:
			featureVectors[sid].append(data)
		else:
			featureVectors[sid] = [data.tolist()]
			groundTruths[sid] = sinfo.getTruthValue(filePath)

# returns: feature vector array (2D), ground truth array (1D)
def collateData(speakerList):
	x = []
	y = []

	for speaker in speakerList:
		if speaker in featureVectors:
			data = featureVectors[speaker]
		else:
			print "ERR: unknown speaker", s
			print featureVectors.keys()
			print groundTruths.keys()
			assert False
		for featureVector in data:
			x.extend(featureVector)
			y.extend(np.full((len(featureVector)), groundTruths[speaker], dtype='int8'))

	return x, y

LoadMFCCFiles(inputPath)