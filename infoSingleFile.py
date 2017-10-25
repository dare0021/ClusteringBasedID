import numpy as np
# Contains information about the files being used
# Separated out since 1) it's long, and 2) the code should change according to what files are used

# if rediculously long, consider making it a np array with a smaller type or even
# a generator instead of a memory dump
lookup = []

def getLastTimestamp(f):
	s = f.readline()
	while (len(s) > 1):
		ts, speaker = s.split()
		s = f.readline()
	# return to file.begin
	f.seek(0, 0)
	# scope? what scope?
	return int(ts)

# retrofit sinfo class
def init(compFilePath, dataLen):
	global lookup

	compFile = open(compFilePath, 'r')
	# GET LASTTIMESTAMP FROM END OF FILE
	multiplier = float(dataLen) / getLastTimestamp(compFile)
	s = compFile.readline()
	lastTime = 0
	while len(s) > 1:
		time, speaker = s.split()
		time = int(time)
		count = round(multiplier * (time - lastTime))
		if speaker == 'M':
			speaker = 0
		elif speaker == 'C':
			speaker = 1
		else:
			print "ERR: Invalid speaker: " + speaker
			assert False
		lookup.extend(np.repeat(speaker, count))
		s = compFile.readline()
		lastTime = time
	compFile.close()
	# rounding errors
	if dataLen > len(lookup):
		diff = dataLen - len(lookup)
		for i in range(diff):
			lookup.append(lookup[-1])
	elif dataLen < len(lookup):
		diff = len(lookup) - dataLen
		lookup = lookup[:len(lookup)-diff]
	assert len(lookup) == dataLen

# returns whether the file path is for a male or female speaker
# 0: Adult
# 1: Child
def getTruthValues():
	return lookup