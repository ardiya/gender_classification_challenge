from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

clf = tree.DecisionTreeClassifier()

## CHALLENGE - create 3 more classifiers...
clfs = {
	"Decision Tree": clf,
	"SVM": SVC(),
	"KNN": KNeighborsClassifier(5),
	"MLP":  MLPClassifier(shuffle=True, solver='lbfgs', random_state=0),
	"AdaBoost": AdaBoostClassifier()
}

#[height, weight, shoe_size]
X = [[190, 86, 10],
 [183, 78, 11],
 [178, 70, 10],
 [180, 68, 9],
 [178, 76, 10],
 [178, 83, 10],
 [175, 68, 10],
 [188, 84, 5],
 [173, 77, 9],
 [173, 70, 10],
 [180, 76, 38],
 [183, 83, 12],
 [178, 72, 10],
 [178, 80, 46],
 [175, 65, 10],
 [183, 73, 40]]
X += [[165, 54, 7],
 [165, 56, 8],
 [160, 60, 7],
 [180, 58, 9],
 [163, 48, 6],
 [170, 52, 8],
 [160, 58, 8],
 [163, 52, 7],
 [168, 54, 9],
 [168, 54, 8],
 [170, 59, 9],
 [155, 51, 6],
 [155, 54, 8],
 [158, 52, 7],
 [160, 52, 8],
 [157, 49, 7],
 [178, 64, 10],
 [160, 54, 7],
 [173, 55, 8],
 [152, 49, 7],
 [173, 57, 8],
 [168, 53, 32],
 [168, 57, 7],
 [160, 60, 8]]

Y = ['male'] * 16 + ['female'] * 24

#CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
for name, c in clfs.items():
	c.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])

#CHALLENGE compare their reusults and print the best one!
top = ('', -1)
for name, c in clfs.items():
	acc = accuracy_score(Y, c.predict(X))
	print(name, "achieve accuracy of:", acc)
	if top[1] < acc: top = (name, acc)
		
print("The highest accuracy is", top[0], "with accuracy of", top[1])

# Output: 
# SVM achieve accuracy of: 1.0
# Decision Tree achieve accuracy of: 1.0
# AdaBoost achieve accuracy of: 1.0
# KNN achieve accuracy of: 0.975
# MLP achieve accuracy of: 1.0
# The highest accuracy is SVM with accuracy of 1.0