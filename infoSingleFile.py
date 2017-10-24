# Contains information about the files being used
# Separated out since 1) it's long, and 2) the code should change according to what files are used

# if rediculously long, consider making it a np array with a smaller type or even
# a generator instead of a memory dump
lookup = []

# retrofit sinfo class
def init(args):
	compFile = open(args['compFile'], 'r')
	dataLen = args['dataLen']
	# GET LASTTIMESTAMP FROM END OF FILE
	multiplier = float(dataLen) / lastTimestamp
	s = compFile.readline()
	lastTime = 0
	while len(s) > 1:
		time, speaker = s.split()
		# ROUNDING!
		count = int(multiplier * (time - lastTime))
		# Create $count long array that countains that many number of 0 or 1
		# extend, not append, to lookup
		s = compFile.readline()
	# add an extra cell for rounding errors?
	# but wont there be more accumulative errors than just 1?
	compFile.close()

# returns whether the file path is for a male or female speaker
# 0: Adult
# 1: Child
def getTruthValue(time)
	return lookup[time]

def getSpeakerID(time):
	retval = ['A', 'C']
	return retval[getTruthValue(time)]

def getSIDKeyType():
	return "string"

def getNbClasses():
	return 2