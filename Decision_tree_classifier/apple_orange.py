from sklearn import tree

feature = [[190,0],[200,0],[210,1],[220,1]]
label = ["Apple","Apple","orange","orange"]

clf = tree.DecisionTreeClassifier()
trained = clf.fit(feature,label)
result = trained.predict([[200,1]])

print (result)

