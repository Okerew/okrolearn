from src.okrolearn.okrolearn import Tensor, np


def test_decision_tree():
    print("Testing Decision Tree...")
    X = Tensor(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    y = np.array([0, 0, 1, 1])

    predictions, tree = X.decision_tree(y, max_depth=2)
    assert predictions.data.shape == (4,), "Decision tree prediction shape mismatch"

    print("Decision Tree test passed!")


def test_gradient_boosting():
    print("Testing Gradient Boosting...")
    X = Tensor(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    y = np.array([0, 0, 1, 1])

    predictions, trees = X.gradient_boosting(y, n_estimators=10)
    assert predictions.data.shape == (4,), "Gradient boosting prediction shape mismatch"
    assert len(trees) == 10, "Incorrect number of trees in gradient boosting"

    print("Gradient Boosting test passed!")


def test_metrics():
    print("Testing metrics...")
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = Tensor(np.array([0, 2, 1, 0, 1, 1]))

    cm = y_pred.confusion_matrix(y_true)
    assert cm.data.shape == (3, 3), "Confusion matrix shape mismatch"

    f1 = y_pred.f1_score(y_true)
    assert f1.data.shape == (3,), "F1 score shape mismatch"

    recall = y_pred.recall(y_true)
    assert recall.data.shape == (3,), "Recall shape mismatch"

    precision = y_pred.precision(y_true)
    assert precision.data.shape == (3,), "Precision shape mismatch"

    print("Metrics tests passed!")


def test_bagging():
    print("Testing Bagging...")
    X = Tensor(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    y = np.array([0, 0, 1, 1])

    predictions, trees = X.bagging(y, n_estimators=5)
    assert predictions.data.shape == (4,), "Bagging prediction shape mismatch"
    assert len(trees) == 5, "Incorrect number of trees in bagging"

    print("Bagging test passed!")


def run_all_tests():
    test_decision_tree()
    test_gradient_boosting()
    test_metrics()
    test_bagging()
    print("All tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
