import xgboost
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import mlflow
import numpy as np
from collections import Counter
from typing import Generator, Dict
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier






def classifier_eval(
    classifier_name: str,
    features: np.ndarray,
    labels: np.ndarray,
):

    classifier_name = classifier_name.lower()
    
    if classifier_name == "xgboost":
        return xgboost_eval(features, labels)
    elif classifier_name == "mlp":
        return mlp_eval(features, labels)
    elif classifier_name == "svc":
        return svc_eval(features, labels)
    else:
        raise ValueError(f"unknown classifier name: {classifier_name}")


def svc_eval(features, labels):
    """
    Perform 5-fold cross validation eval with a SVC.
    """

    for split_id, ((train_x, train_y), (test_x, test_y)) in enumerate(get_splits(features, labels)):

        # train model
        svc = SVC()
        svc.fit(train_x, train_y)

        # test
        test_preds = svc.predict(test_x)

        # metrics
        handle_preds(test_y, test_preds, split_id)

    


def mlp_eval(features, labels):
    """
    Perform 5-fold cross validation eval with an MLP classifier. 
    """
    
    num_classes = len(np.unique(labels))
    feature_dim = features.shape[1]

    for split_id, ((train_x, train_y), (test_x, test_y)) in enumerate(get_splits(features, labels)):

        # train model
        mlp = MLPClassifier(
            hidden_layer_sizes=(feature_dim, 100, num_classes)
        )
        mlp.fit(train_x, train_y)


        # test
        test_preds = mlp.predict(test_x)

        # metrics
        handle_preds(test_y, test_preds, split_id)


def xgboost_eval(features, labels):
    """
    Perform 5-fold cross validation with the XGBoost classifier.
    """

    for split_id, ((train_x, train_y), (test_x, test_y)) in enumerate(get_splits(features, labels)):

        # put into xgboost preferred format
        dtrain = xgboost.DMatrix(train_x, train_y)
        dtest = xgboost.DMatrix(test_x, test_y)

        # train model
        param = {
            "max_depth": 2,
            "eta": 1,
            "objective": "multi:softprob",
            "num_class": len(np.unique(labels))
        }
        model = xgboost.train(
            param,
            dtrain,
            num_boost_round=2,
        )

        # test
        test_preds = np.argmax(model.predict(dtest), axis=1)

        # metrics
        handle_preds(test_y, test_preds, split_id)


"""
Utils
"""

def get_splits(features, labels) -> Generator:
    """
    Return data splits in a generator 
    """

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(features):
        x_train, y_train = features[train_index], labels[train_index]
        x_test, y_test = features[test_index], labels[test_index]

        yield (x_train, y_train), (x_test, y_test)


def get_metrics(y_true, y_pred):
    return {
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "acc": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "prec": precision_score(y_true, y_pred, average="weighted")
    }


def handle_preds(y_true: np.ndarray, y_pred:np.ndarray, split_id: int) -> None:
    """
    Calculate metrics and log them to MLFlow 
    """

    # get metrics
    metrics = get_metrics(y_true, y_pred)

    # log metrics
    for metric_name, metric_value in metrics.items():
        metric_key = f"{metric_name}_split_{split_id}"
        mlflow.log_metric(metric_key, metric_value)

    # count labels and log the counts
    count_dict = dict(Counter(y_true))
    for class_index, class_count in count_dict.items():
        mlflow.log_metric(f"n_class_{class_index}", class_count)

