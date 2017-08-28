# too much clutter to keep in run.py, which is otherwise a logics file

from threading import Thread, BoundedSemaphore
import sklearn 
import numpy as np
from sklearn import neighbors, svm, cluster
import speakerInfo as sinfo
num_threads_sema = None
threadFunction = None
# default 200, in MB
ram_usage = 1024

class ModelSettings:
	def __init__(self, i, paaFunction, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, speakerName, args=None):
		self.i = i
		self.paaFunction = paaFunction
		self.trainFeatureVector = trainFeatureVector
		self.testFeatureVector = testFeatureVector
		self.trainTruthVector = trainTruthVector
		self.testTruthVector = testTruthVector
		self.speakerName = speakerName
		self.args = args

def init(num_threads_sema_input, functionToRun):
	global num_threads_sema
	global threadFunction
	num_threads_sema = num_threads_sema_input
	threadFunction = functionToRun

def runModel(modelFunc, tag, ms):
	global num_threads_sema
	num_threads_sema.acquire()
	p = Thread(target=threadFunction, args=(modelFunc, tag, ms))
	p.start()

def model_KNN():
	print 'Running KNN'
	return sklearn.neighbors.KNeighborsClassifier(n_neighbors=sinfo.getNbClasses())

# data set too small
def model_RNC():
	print 'Running RNC'
	return sklearn.neighbors.RadiusNeighborsClassifier()

def model_SVM_linear():
	print 'Running SVM Linear'
	return sklearn.svm.SVC(kernel='linear', cache_size=ram_usage)

def model_SVM_poly():
	print 'Running SVM Poly'
	return sklearn.svm.SVC(kernel='poly', cache_size=ram_usage)

def factory_SVM_rbf(gamma=1.0/sinfo.getNbClasses(), shrinking=True, tol=1e-3, c=1):
	return (gamma, shrinking, tol, c)

# Radial Basis Function
def model_SVM_rbf(args = factory_SVM_rbf()):
	print 'Running SVM RBF'
	return sklearn.svm.SVC(kernel='rbf', gamma=args[0], shrinking=args[1], tol=args[2], C=args[3], cache_size=ram_usage)

def model_SVM_sigmoid():
	print 'Running SVM Sigmoid'
	return sklearn.svm.SVC(kernel='sigmoid', cache_size=ram_usage)

# not enough memory
def model_Spectral():
	print 'Running Spectral Clustering'
	return sklearn.cluster.SpectralClustering(n_clusters=sinfo.getNbClasses())

# not enough memory
def model_MiniK():
	print 'Running Mini K'
	return sklearn.cluster.MiniBatchKMeans(n_clusters=sinfo.getNbClasses())

def model_ACWard():
	print 'Running AC Ward'
	return sklearn.cluster.AgglomerativeClustering(n_clusters=sinfo.getNbClasses(), linkage='ward')

def model_ACComplete():
	print 'Running AC Complete'
	return sklearn.cluster.AgglomerativeClustering(n_clusters=sinfo.getNbClasses(), linkage='complete')

def model_ACAverage():
	print 'Running AC Average'
	return sklearn.cluster.AgglomerativeClustering(n_clusters=sinfo.getNbClasses(), linkage='average')

# clustering class incompatible with classifiers
# clustering is deterministic, non-ML? Doesn't seem to use training at all.
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
def model_Birch():
	print 'Running Birch'
	return sklearn.cluster.Birch(n_clusters=sinfo.getNbClasses())

def runAllModels(ms):
	runModel(model_KNN, 'PAA_' + str(ms.paaFunction) + '_KNN_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_RNC, 'PAA_' + str(ms.paaFunction) + '_RNC_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_linear, 'PAA_' + str(ms.paaFunction) + '_SVM_Linear_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_poly, 'PAA_' + str(ms.paaFunction) + '_SVM_Poly_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_rbf, 'PAA_' + str(ms.paaFunction) + '_SVM_RBF_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_sigmoid, 'PAA_' + str(ms.paaFunction) + '_SVM_Sigmoid_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_Spectral, 'PAA_' + str(ms.paaFunction) + '_SpectralClustering_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_MiniK, 'PAA_' + str(ms.paaFunction) + '_MiniK_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_ACWard, 'PAA_' + str(ms.paaFunction) + '_ACWard_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_ACAverage, 'PAA_' + str(ms.paaFunction) + '_ACAvg_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_ACComplete, 'PAA_' + str(ms.paaFunction) + '_ACComplete_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_Birch, 'PAA_' + str(ms.paaFunction) + '_Birch_' + str(ms.i) + '_' + ms.speakerName, ms)

def runAllModelsPAA(ms, windowSize):
	if windowSize >= 300:
		runModel(model_Spectral, 'PAA_' + str(ms.paaFunction) + '_SpectralClustering_' + str(ms.i) + '_' + ms.speakerName, ms)
		runModel(model_MiniK, 'PAA_' + str(ms.paaFunction) + '_MiniK_' + str(ms.i) + '_' + ms.speakerName, ms)	
	if windowSize >= 50:
		runModel(model_ACWard, 'PAA_' + str(ms.paaFunction) + '_ACWard_' + str(ms.i) + '_' + ms.speakerName, ms)
		runModel(model_ACAverage, 'PAA_' + str(ms.paaFunction) + '_ACAvg_' + str(ms.i) + '_' + ms.speakerName, ms)
		runModel(model_ACComplete, 'PAA_' + str(ms.paaFunction) + '_ACComplete_' + str(ms.i) + '_' + ms.speakerName, ms)	
	runModel(model_KNN, 'PAA_' + str(ms.paaFunction) + '_KNN_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_linear, 'PAA_' + str(ms.paaFunction) + '_SVM_Linear_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_poly, 'PAA_' + str(ms.paaFunction) + '_SVM_Poly_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_rbf, 'PAA_' + str(ms.paaFunction) + '_SVM_RBF_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_sigmoid, 'PAA_' + str(ms.paaFunction) + '_SVM_Sigmoid_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_Birch, 'PAA_' + str(ms.paaFunction) + '_Birch_' + str(ms.i) + '_' + ms.speakerName, ms)

def runAllModelsMFCC(ms):
	runModel(model_KNN, 'MFCC_' + str(ms.paaFunction) + '_KNN_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_linear, 'MFCC_' + str(ms.paaFunction) + '_SVM_Linear_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_poly, 'MFCC_' + str(ms.paaFunction) + '_SVM_Poly_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_rbf, 'MFCC_' + str(ms.paaFunction) + '_SVM_RBF_' + str(ms.i) + '_' + ms.speakerName, ms)
	runModel(model_SVM_sigmoid, 'PAA_' + str(ms.paaFunction) + '_SVM_Sigmoid_' + str(ms.i) + '_' + ms.speakerName, ms)

def runRBFvariantsGamma(ms, gammaMin, gammaMax, gammaStep):
	for heuristicsOn in [True]:
		print 'heuristics', heuristicsOn
		for gamma in np.arange(gammaMin, gammaMax, gammaStep):
			print 'gamma', gamma
			ms.args = factory_SVM_rbf(gamma, heuristicsOn)
			runModel(model_SVM_rbf, 'MFCC_' + str(ms.paaFunction) + '_SVM_RBF_g_' + str(gamma) + '_H_' + str(heuristicsOn) + '_' + str(ms.i) + '_' + ms.speakerName, ms)

def runRBFvariantsCList(ms, cList, gamma):
	for heuristicsOn in [True]:
		print 'heuristics', heuristicsOn
		for c in cList:
			print 'c', c
			ms.args = factory_SVM_rbf(gamma, heuristicsOn, c)
			runModel(model_SVM_rbf, 'MFCC_' + str(ms.paaFunction) + '_SVM_RBF_g_' + str(gamma) + '_c_' + str(c) + '_H_' + str(heuristicsOn) + '_' + str(ms.i) + '_' + ms.speakerName, ms)

def runRBFvariants2DList(ms, cList, gammaList):
	for heuristicsOn in [True]:
		print 'heuristics', heuristicsOn
		for c in cList:
			for gamma in gammaList:
				print 'c', c, 'g', gamma
				ms.args = factory_SVM_rbf(gamma, heuristicsOn, c)
				runModel(model_SVM_rbf, 'MFCC_' + str(ms.paaFunction) + '_SVM_RBF_g_' + str(gamma) + '_c_' + str(c) + '_H_' + str(heuristicsOn) + '_' + str(ms.i) + '_' + ms.speakerName, ms)