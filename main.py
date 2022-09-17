import mlflow
from models.models import *
from preprocessing.data_preprocessing import *
from utils.utils import *

if __name__ == "__main__":
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Zindi_experiment")
    train = pd.read_csv('../umojahack-africa-2022-beginner-challenge/train.csv', parse_dates=['Datetime'])
    inference = pd.read_csv('../umojahack-africa-2022-beginner-challenge/test.csv', parse_dates=['Datetime'])
    train = date_processing(train)
    X = train.drop('Offset_fault', axis='columns')
    y = train.iloc[:,5]
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2, stratify=y)
    X_train = missing_values_processing(X_train)
    X_train = outlier_processing(X_train)
    with mlflow.start_run():
        classifier, train_accuracy, test_accuracy = svm_classifier(X_train, X_test, y_train, y_test)
        y_pred = classifier.predict(y_test)
        f1, auc = calculate_metrics(classifier, X_test, y_test, y_pred)
        params = {"kernel":'rbf', "random_state": 42}
        metrics = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                "f1": f1, "auc": auc}
        roc = plot_auc_roc(y_test, y_pred)
        conf_matrix = confusion_matrix(classifier, X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_figure(roc, "svm_roc.png")
        mlflow.log_figure(conf_matrix, "svm_confusion_matrix.png")
        mlflow.sklearn.log_model(classifier, artifact_path="models")
        mlflow.register_model(classifier, "SVC")
    
    with mlflow.start_run():
        clf, train_accuracy, test_accuracy = lr_classifier(X_train, X_test, y_train, y_test)
        y_pred = clf.predict(y_test)
        f1, auc = calculate_metrics(clf, X_test, y_test, y_pred)
        params = {"random_state": 42, "C": 1e5}
        metrics = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                "f1": f1, "auc": auc}
        roc = plot_auc_roc(y_test, y_pred)
        conf_matrix = confusion_matrix(clf, X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_figure(roc, "lr_roc.png")
        mlflow.log_figure(conf_matrix, "lr_confusion_matrix.png")
        mlflow.sklearn.log_model(clf, artifact_path="models")
        mlflow.register_model(clf, "LogisticRegression")
        
    with mlflow.start_run():
        tree_clf, train_accuracy, test_accuracy = tree_classifier(X_train, X_test, y_train, y_test)
        y_pred = tree_clf.predict(y_test)
        f1, auc = calculate_metrics(tree_clf, X_test, y_test, y_pred)
        params = {"max_depth": 10}
        metrics = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                "f1": f1, "auc": auc}
        roc = plot_auc_roc(y_test, y_pred)
        conf_matrix = confusion_matrix(tree_clf, X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_figure(roc, "tree_roc.png")
        mlflow.log_figure(conf_matrix, "tree_confusion_matrix.png")
        mlflow.sklearn.log_model(tree_clf, artifact_path="models")
        mlflow.register_model(tree_clf, "decision_trees")
        
    with mlflow.start_run():
        for_clf, train_accuracy, test_accuracy = forest_classifier(X_train, X_test, y_train, y_test)
        y_pred = for_clf.predict(y_test)
        f1, auc = calculate_metrics(for_clf, X_test, y_test, y_pred)
        metrics = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                "f1": f1, "auc": auc}
        roc = plot_auc_roc(y_test, y_pred)
        conf_matrix = confusion_matrix(for_clf, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.log_figure(roc, "random_forest_roc.png")
        mlflow.log_figure(conf_matrix, "random_forest_confusion_matrix.png")
        mlflow.sklearn.log_model(for_clf, artifact_path="models")
        mlflow.register_model(for_clf, "random_forest")
        
    with mlflow.start_run():
        sgd_clf, train_accuracy, test_accuracy = sgd_classifier(X_train, X_test, y_train, y_test)
        y_pred = sgd_clf.predict(y_test)
        f1, auc = calculate_metrics(sgd_clf, X_test, y_test, y_pred)
        params = {"max_iter": 1000, "tol": 1e-3}
        metrics = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                "f1": f1, "auc": auc}
        roc = plot_auc_roc(y_test, y_pred)
        conf_matrix = confusion_matrix(sgd_clf, X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_figure(roc, "sgd_roc.png")
        mlflow.log_figure(conf_matrix, "sgd_confusion_matrix.png")
        mlflow.sklearn.log_model(sgd_clf, artifact_path="models")
        mlflow.register_model(sgd_clf, "sgd")