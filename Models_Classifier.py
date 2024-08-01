
def c45():
    ### Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None)
    return clf


def cart():
    ### Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, max_features='sqrt')
    return clf


def knn():
    ### KNeighbor
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=6, weights='distance',metric=1, n_jobs=8)
    return clf


def lrc():
    ### Logistic Regression Classifier    ###penalty='l2'
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')
    return clf


def rf10(n_estimators=10):
    ### Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_estimators)
    
    return clf

def rf20(n_estimators=20):
    ### Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_estimators)
    
    return clf

def rf30(n_estimators=30):
    ### Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', bootstrap=True, n_jobs=8, class_weight=None, min_samples_leaf=1, max_features=None)
    
    return clf


def gbdt(n_estimators=200):
    ### GBDT(Gradient Boosting Decision Tree) Classifier
    ### n_estimators = 200
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, subsample=1, n_estimators=n_estimators, criterion='friedman_mse')
    
    return clf


def AdaBoost():
    ###AdaBoost Classifier
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=60, learning_rate=0.1)
    
    return clf


def gnb():
    ### GaussianNB
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB(priors=None, var_smoothing=1e-09) # change priors can improve (add Probability of normal)
    
    return clf


def lda():
    ### Linear Discriminant Analysis
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis(solver='eigen', priors=None) # change priors can improve (add Probability of normal)
    
    return clf


def qda():
    ### Quadratic Discriminant Analysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    clf = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, tol=1e-04)
    
    return clf


def svm(kernel='poly', probability=True):
    ### SVM Classifier
    from sklearn.svm import SVC
    # clf = SVC(kernel='rbf,linear,poly', probability=True)
    clf = SVC(kernel=kernel, probability=probability)
    return clf
