import numpy as np
from PIL import Image as imaging
import os

# validates and compiles diarization file
tfColors = [(225,90,90),(20,230,40),(255,255,255)]
pixelGraphZoom = 5

# The data is in decisecond resolution
# What am I gonna do? Redo the data? Use float?
def timestampToDeciseconds(timestamp):
	retval = int(timestamp[-1])
	timestamp = timestamp[:len(timestamp)-1]
	multiplier = 10
	while len(timestamp) > 2:
		retval += int(timestamp[len(timestamp)-2:]) * multiplier
		timestamp = timestamp[:len(timestamp)-2]
		multiplier *= 60
	if len(timestamp) > 0:
		retval += int(timestamp) * multiplier
	return retval

def run(inputPath):
	f = open(inputPath, 'r')
	outBuff = []
	s = f.readline()
	lastTime = 0
	lastSpeaker = None
	while len(s) > 1:
		timestamp, speaker = s.split()
		deci = timestampToDeciseconds(timestamp)
		if lastTime >= deci:
			print "ERR: time backtracking:", lastTime, deci, timestamp
		if lastSpeaker == speaker:
			print "ERR: no speaker change", speaker
		outBuff.append([deci, speaker])
		lastTime = deci
		s = f.readline()
	f.close()
	return outBuff

def saveOutput(outputPath, outBuff):
	writeBuff = ""
	for i in outBuff:
		deci, speaker = i
		writeBuff += str(deci) + '\t' + speaker + '\n'
	fo = open(outputPath, 'w')
	fo.write(writeBuff)
	fo.close()

# VERY BROKEN
def generatePixelGraph(numList, colorList, filePath):
	def numToColor(num):
		if num == 'M':
			num = 0
		elif num == 'C':
			num = 1
		else:
			print "ERR: invalid input:", num
			assert False
		return colorList[num]

	pxs = []
	for i in numList:
		for j in range(i[0]):
			pxs.append(numToColor(i[1]))
	pxs = np.array([pxs])
	img = imaging.fromarray(pxs, 'RGB')
	img = img.resize((len(pxs) * pixelGraphZoom, pixelGraphZoom))
	img.save(filePath)

# check last tag != this tag
# check last time < this time
# generate truth graph?
# generate sinfo file parameters?

inputPath = 'joo proc pass 3.wav.diarization'
outputPath = inputPath + '.comp'
output = run(inputPath)
saveOutput(outputPath, output)
generatePixelGraph(output, tfColors, outputPath + '.png')