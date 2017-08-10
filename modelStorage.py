# too much clutter to keep in run.py, which is otherwise a logics file

from threading import Thread, BoundedSemaphore
import sklearn 
from sklearn import neighbors, svm, cluster
import speakerInfo as sinfo
num_threads_sema = None
threadFunction = None
paaFunction = -1

def init(num_threads_sema_input, functionToRun):
	global num_threads_sema
	global threadFunction
	num_threads_sema = num_threads_sema_input
	threadFunction = functionToRun

def runModel(modelFunc, tag, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector):
	global num_threads_sema
	num_threads_sema.acquire()
	p = Thread(target=threadFunction, args=(modelFunc, tag, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector))
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
	return sklearn.svm.SVC(kernel='linear')

def model_SVM_poly():
	print 'Running SVM Poly'
	return sklearn.svm.SVC(kernel='poly')

# Radial Basis Function
def model_SVM_rbf():
	print 'Running SVM RBF'
	return sklearn.svm.SVC(kernel='rbf')

def model_SVM_sigmoid():
	print 'Running SVM Sigmoid'
	return sklearn.svm.SVC(kernel='sigmoid')

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

def runAllModels(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, paaFunctionName):
	runModel(model_KNN, 'PAA_' + str(paaFunction) + '_KNN_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_RNC, 'PAA_' + str(paaFunction) + '_RNC_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_linear, 'PAA_' + str(paaFunction) + '_SVM_Linear_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_poly, 'PAA_' + str(paaFunction) + '_SVM_Poly_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_rbf, 'PAA_' + str(paaFunction) + '_SVM_RBF_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_sigmoid, 'PAA_' + str(paaFunction) + '_SVM_Sigmoid_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_Spectral, 'PAA_' + str(paaFunction) + '_SpectralClustering_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_MiniK, 'PAA_' + str(paaFunction) + '_MiniK_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACWard, 'PAA_' + str(paaFunction) + '_ACWard_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACAverage, 'PAA_' + str(paaFunction) + '_ACAvg_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_ACComplete, 'PAA_' + str(paaFunction) + '_ACComplete_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_Birch, 'PAA_' + str(paaFunction) + '_Birch_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)

def runAllModelsPAA(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, windowSize, paaFunctionName):
	if windowSize >= 300:
		runModel(model_Spectral, 'PAA_' + str(paaFunction) + '_SpectralClustering_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
		runModel(model_MiniK, 'PAA_' + str(paaFunction) + '_MiniK_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)	
	if windowSize >= 50:
		runModel(model_ACWard, 'PAA_' + str(paaFunction) + '_ACWard_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
		runModel(model_ACAverage, 'PAA_' + str(paaFunction) + '_ACAvg_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
		runModel(model_ACComplete, 'PAA_' + str(paaFunction) + '_ACComplete_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)	
	runModel(model_KNN, 'PAA_' + str(paaFunction) + '_KNN_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_linear, 'PAA_' + str(paaFunction) + '_SVM_Linear_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_poly, 'PAA_' + str(paaFunction) + '_SVM_Poly_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_rbf, 'PAA_' + str(paaFunction) + '_SVM_RBF_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_sigmoid, 'PAA_' + str(paaFunction) + '_SVM_Sigmoid_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_Birch, 'PAA_' + str(paaFunction) + '_Birch_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)

def runAllModelsMFCC(i, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector, paaFunctionName):
	runModel(model_KNN, 'MFCC_' + str(paaFunction) + '_KNN_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_linear, 'MFCC_' + str(paaFunction) + '_SVM_Linear_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_poly, 'MFCC_' + str(paaFunction) + '_SVM_Poly_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_rbf, 'MFCC_' + str(paaFunction) + '_SVM_RBF_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)
	runModel(model_SVM_sigmoid, 'PAA_' + str(paaFunction) + '_SVM_Sigmoid_' + str(i) + '_' + paaFunctionName, trainFeatureVector, testFeatureVector, trainTruthVector, testTruthVector)