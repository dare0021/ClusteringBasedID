# validates and compiles diarization file
tfColors = [(225,90,90),(20,230,40),(255,255,255)]

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
		assert lastTime < deci
		assert lastSpeaker != speaker
		outBuff.append([deci, speaker])
		s = f.readline()
	f.close()
	return outBuff

def saveOutput(outputPath, outBuff):
	writeBuff = ""
	for i in outBuff:
		deci, speaker = outBuff
		writeBuff += str(deci) + '\t' + speaker + '\n'
	fo = open(outputPath, 'w')
	fo.write(writeBuff)
	fo.close()

def generatePixelGraph(numList, colorList, filePath):
	def numToColor(num):
		return colorList[num]

	numList = np.resize(1, len(numList))
	pxs = np.array([map(numToColor, i) for i in numList], dtype='int8')
	img = imaging.fromarray(pxs, 'RGB')
	img = img.resize((len(pxs) * pixelGraphZoom, pixelGraphZoom))
	assert not os.path.isfile(filePath)
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