import numpy
from sklearn import tree
from sklearn.datasets import load_iris

#creating object for dataset
iris = load_iris()
#creating object for classifier algorithm
clf = tree.DecisionTreeClassifier()

#creating a training data set
training_data = numpy.delete(iris.data,[0,50,100],axis=0) #training_Data/feature
training_target = numpy.delete(iris.target,[0,50,100])    #training_target/label

test_data = iris.data[[0,50,100]]

trained = clf.fit(training_data,training_target)
result = trained.predict(test_data[[0]])

print (iris.target_names[result[0]])
