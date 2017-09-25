import os

inputPath = "/media/jkih/b6988675-1154-47d9-9d37-4a80b771f7fe/new/sukwoo/Sphinx SVM_RBF g search 0.001 0.1 0.001 non-clairvoyant/"

class Result:
	# param order same as file name
	def __init__(self, accmean, f1mean, rawString):
		self.accmean = accmean
		self.f1mean = f1mean
		self.rawString = rawString

	def __str__(self):
		return self.rawString

def parseKVP(s):
	return (s[:s.find(':')].strip(), float(s[s.find(':')+2:]))

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
	return Result(accmean, f1mean, rawString)

def loadFiles(parentDir):
	results = []
	i = 0
	for di in [x[0] for x in os.walk(parentDir) if 'inferred' in x[0]]:
		i += 1
		# results.append(readFile(di + "stats/summary.txt"))
	print i
	return results

def printMax(results, outputPath):
	accmeanmax = results[0]
	f1meanmax = results[0]
	for result in results:
		if result.accmean > accmeanmax.accmean:
			accmeanmax = result
		if result.f1mean > f1meanmax.f1mean:
			f1meanmax = result
	print 'acc max:'
	print accmeanmax
	print ' '
	print 'f1 max:'
	print f1meanmax

	f = open(outputPath, 'w')
	f.write(str(accmeanmax))
	f.write('\n\n')
	f.write(str(f1meanmax))
	f.close()
	# find accmean max
	# find f1mean max
	# compose string of the max and their file contents as saved in the Result object
	# print and save to file

