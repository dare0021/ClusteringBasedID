import os

class Result:
	# param order same as file name
	def __init__(self, accmean, f1mean, rawString, identifier):
		self.accmean = accmean
		self.f1mean = f1mean
		self.rawString = rawString
		self.identifier = identifier

	def __str__(self):
		return self.identifier + '\n' + self.rawString

def parseKVP(s):
	return (s[:s.find(':')].strip(), float(s[s.find(':')+2:]))

def parseMaxLine(s):
	return (s[s.find("'")+1:s.rfind("'")], float(s[s.find('(')+1:s.find(',')]))

def readFile(path):
	f = open(path, 'r')
	rawString = f.readline()
	garbage, accmean = parseKVP(rawString)
	rawString += f.readline()
	rawString += f.readline()
	rawString += f.readline()
	s = f.readline()
	garbage, f1mean = parseKVP(s)
	f.close()
	rawString += s
	path = path[:path.rfind('/')]
	path = path[:path.rfind('/')]
	path = path[path.rfind('/')+1:]
	return Result(accmean, f1mean, rawString, path)

def loadFiles(parentDir):
	overall = []
	maxByModel = []
	maxIDByModel = []
	for di in [x[0] for x in os.walk(parentDir) if ('inferred' in x[0] and 'stats' in x[0])]:
		print di
		summaryFile = di + "/summary.txt"
		overall.append(readFile(summaryFile))
		maxesIter, maxIDsIter = getLastMaxBlock(summaryFile, di[di.rfind('inferred')+9:di.rfind('/')])
		maxByModel.append(maxesIter)
		maxIDByModel.append(maxIDsIter)
	return overall, maxByModel, maxIDByModel

# sure using a generator is more safe
# but meh nothing about this function is safe
def getLastMaxBlock(path, fID):
	maxes = [0,0,0,0]
	maxIDs = [0,0,0,0]
	backoff = 512
	f = open(path, 'r')

	f.seek(-1 * backoff, 2)
	s = f.readline()
	while not ('AccMax' in s):
		s = f.readline()
	for i in range(4):
		k, v = parseMaxLine(s)
		maxes[i] = v
		maxIDs[i] = k + fID
		s = f.readline()

	f.close()
	return maxes, maxIDs

def printAndSaveMaxLine(f, key, val, ID):
	s = key + " : (" + str(val) + ', ' + ID + "')"
	print s
	f.write(s + '\n')

def printMax(results, outputPath):
	overall, maxByModel, maxIDByModel = results

	accmeanmax = overall[0]
	f1meanmax = overall[0]
	for result in overall:
		if result.accmean > accmeanmax.accmean:
			accmeanmax = result
		if result.f1mean > f1meanmax.f1mean:
			f1meanmax = result
	print 'acc max:'
	print accmeanmax
	print ' '
	print 'f1 max:'
	print f1meanmax

	maxes = [0,0,0,0]
	maxIDs = [0,0,0,0]
	for i in range(len(maxByModel)):
		maxesIter = maxByModel[i]
		maxIDsIter = maxIDByModel[i]
		for i in range(len(maxes)):
			if maxesIter[i] > maxes[i]:
				maxes[i] = maxesIter[i]
				maxIDs[i] = maxIDsIter[i]

	f = open(outputPath, 'w')

	f.write(str(accmeanmax))
	f.write('\n\n')
	f.write(str(f1meanmax))
	f.write('\n\n')

	printAndSaveMaxLine(f, 'AccMax by Mean  ', maxes[0], maxIDs[0])	
	printAndSaveMaxLine(f, 'AccMax by Median', maxes[1], maxIDs[1])	
	printAndSaveMaxLine(f, 'F1 Max by Mean  ', maxes[2], maxIDs[2])	
	printAndSaveMaxLine(f, 'F1 Max by Median', maxes[3], maxIDs[3])	

	f.close()

inputPath = "/media/jkih/b6988675-1154-47d9-9d37-4a80b771f7fe/new/sukwoo/shortsegs randomforest/10 0.1 avg 10forests/"
# inputPath = '/media/jkih/b6988675-1154-47d9-9d37-4a80b771f7fe/new/sukwoo/shortsegs archive/f 0 0.4 0.01 d 16/'
results = loadFiles(inputPath)
printMax(results, inputPath + 'inf stats.txt')