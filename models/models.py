from sklearn import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

def svm_classifier(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel = 'rbf', random_state = 42)
    classifier.fit(X_train, y_train)
    train_accuracy = classifier.score(X_train,y_train)
    test_accuracy = classifier.score(X_test,y_test)
    return classifier, train_accuracy, test_accuracy

def lr_classifier(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=1e5, random_state=42)
    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train,y_train)
    test_accuracy = clf.score(X_test,y_test)
    return clf, train_accuracy, test_accuracy

def tree_classifier(X_train, X_test, y_train, y_test):
    tree_clf = DecisionTreeClassifier(max_depth = 10)
    tree_clf.fit(X_train,y_train)
    train_accuracy = tree_clf.score(X_train,y_train)
    test_accuracy = tree_clf.score(X_test,y_test)
    return tree_clf, train_accuracy, test_accuracy

def forest_classifier(X_train, X_test, y_train, y_test):
    for_clf = RandomForestClassifier()
    for_clf.fit(X_train,y_train)
    train_accuracy = for_clf.score(X_train,y_train)
    test_accuracy = for_clf.score(X_test,y_test)
    return for_clf, train_accuracy, test_accuracy

def sgd_classifier(X_train, X_test, y_train, y_test):
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train, y_train)
    train_accuracy = sgd_clf.score(X_train,y_train)
    test_accuracy = sgd_clf.score(X_test,y_test)
    return sgd_clf, train_accuracy, test_accuracy

