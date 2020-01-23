from sklearn import tree
import numpy
from sklearn.datasets import load_iris
new_data = numpy.delete(load_iris().data,[0,50,100],axis=0)
new_target = numpy.delete(load_iris().target,[0,50,100])
check_data = load_iris().data[[0,50,100]]
check_target = load_iris().target[[0,50,100]]
clf = tree.DecissionTreeClassifer()
trained = clf.fit(new_data,new_target)
result = trained.predict(check_data[[0]])
print (result)

