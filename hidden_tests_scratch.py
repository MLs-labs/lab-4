"""
Hidden tests for Lab 4 — K-Nearest Neighbors from scratch.

This file must NEVER be included in student repositories.

The tests verify behavior, not the student's exact implementation.
"""

import numpy as np


# ============================================================
# Helpers
# ============================================================

def _passed():
    print("\033[92mAll tests passed!\033[0m")


def _fail(message):
    raise AssertionError(message)


# ============================================================
# 1. Constructor
# ============================================================

def knn_init_test(KNNClassifier):
    """
    Verify that KNNClassifier initializes correctly.
    """

    model = KNNClassifier(k=5)

    # --------------------------------------------------
    # k
    # --------------------------------------------------

    if model.k != 5:
        _fail(
            f"k was not stored correctly: expected 5, got {model.k}"
        )

    # --------------------------------------------------
    # Initial training state
    # --------------------------------------------------

    if not hasattr(model, "X_"):
        _fail("KNNClassifier must define X_.")

    if not hasattr(model, "y_"):
        _fail("KNNClassifier must define y_.")

    if model.X_ is not None:
        _fail("X_ should initially be None.")

    if model.y_ is not None:
        _fail("y_ should initially be None.")

    # --------------------------------------------------
    # Required methods
    # --------------------------------------------------

    required_methods = [
        "fit",
        "predict",
        "score",
        "_euclidean_distance",
        "_get_neighbors",
        "_majority_vote",
    ]

    for method_name in required_methods:
        if not callable(getattr(model, method_name, None)):
            _fail(
                f"KNNClassifier must implement {method_name}()."
            )

    # --------------------------------------------------
    # Invalid k
    # --------------------------------------------------

    for invalid_k in [0, -1]:

        try:
            KNNClassifier(k=invalid_k)
        except ValueError:
            pass
        else:
            _fail(
                f"KNNClassifier should raise ValueError for k={invalid_k}."
            )

    _passed()


# ============================================================
# 2. Fit
# ============================================================

def knn_fit_test(KNNClassifier):
    """
    Verify that fit() stores the training data correctly.
    """

    X_train = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
    ])

    y_train = np.array([
        0,
        0,
        0,
        1,
    ])

    model = KNNClassifier(k=3)

    returned = model.fit(X_train, y_train)

    # --------------------------------------------------
    # fit returns self
    # --------------------------------------------------

    if returned is not model:
        _fail("fit() should return self.")

    # --------------------------------------------------
    # Training data attributes
    # --------------------------------------------------

    if model.X_ is None:
        _fail("fit() must create X_.")

    if model.y_ is None:
        _fail("fit() must create y_.")

    if not isinstance(model.X_, np.ndarray):
        _fail("X_ must be a NumPy array.")

    if not isinstance(model.y_, np.ndarray):
        _fail("y_ must be a NumPy array.")

    if model.X_.shape != X_train.shape:
        _fail(
            f"X_ has wrong shape: "
            f"expected {X_train.shape}, got {model.X_.shape}"
        )

    if model.y_.shape != y_train.shape:
        _fail(
            f"y_ has wrong shape: "
            f"expected {y_train.shape}, got {model.y_.shape}"
        )

    if not np.array_equal(model.X_, X_train):
        _fail("X_ does not match the training data.")

    if not np.array_equal(model.y_, y_train):
        _fail("y_ does not match the training labels.")

    # --------------------------------------------------
    # k cannot exceed training size
    # --------------------------------------------------

    model_small = KNNClassifier(k=10)

    try:
        model_small.fit(X_train, y_train)
    except ValueError:
        pass
    else:
        _fail(
            "fit() should raise ValueError when k is larger "
            "than the number of training samples."
        )

    _passed()


# ============================================================
# 3. Euclidean distance
# ============================================================

def knn_distance_test(KNNClassifier):
    """
    Verify Euclidean distance behavior.
    """

    model = KNNClassifier(k=3)

    # --------------------------------------------------
    # Classic 3-4-5 triangle
    # --------------------------------------------------

    x1 = np.array([0.0, 0.0])
    x2 = np.array([3.0, 4.0])

    result = model._euclidean_distance(x1, x2)

    if not np.isscalar(result):
        _fail(
            "_euclidean_distance() must return a scalar."
        )

    if not np.isfinite(result):
        _fail(
            "_euclidean_distance() returned NaN or infinity."
        )

    if not np.isclose(result, 5.0, atol=1e-8):
        _fail(
            f"Expected distance 5.0, got {result}"
        )

    # --------------------------------------------------
    # Same point
    # --------------------------------------------------

    same = np.array([2.0, -3.0])

    result_same = model._euclidean_distance(
        same,
        same,
    )

    if not np.isclose(result_same, 0.0, atol=1e-8):
        _fail(
            "Distance from a point to itself must be zero."
        )

    # --------------------------------------------------
    # Symmetry
    # --------------------------------------------------

    result_xy = model._euclidean_distance(x1, x2)
    result_yx = model._euclidean_distance(x2, x1)

    if not np.isclose(result_xy, result_yx, atol=1e-8):
        _fail(
            "Euclidean distance must be symmetric."
        )

    # --------------------------------------------------
    # Non-negative
    # --------------------------------------------------

    if result_xy < 0:
        _fail(
            "Euclidean distance cannot be negative."
        )

    _passed()


# ============================================================
# 4. Nearest neighbors
# ============================================================

def knn_neighbors_test(KNNClassifier):
    """
    Verify that _get_neighbors() identifies the correct
    nearest training samples.
    """

    X_train = np.array([
        [0.0, 0.0],   # index 0
        [1.0, 0.0],   # index 1
        [0.0, 1.0],   # index 2
        [5.0, 5.0],   # index 3
        [6.0, 5.0],   # index 4
    ])

    y_train = np.array([
        0,
        0,
        0,
        1,
        1,
    ])

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    query = np.array([0.2, 0.1])

    neighbors = model._get_neighbors(query)

    if not isinstance(neighbors, np.ndarray):
        _fail(
            "_get_neighbors() must return a NumPy array."
        )

    if neighbors.shape != (3,):
        _fail(
            f"Expected 3 neighbor indices, got shape {neighbors.shape}."
        )

    if not np.all(np.isin(neighbors, np.arange(len(X_train)))):
        _fail(
            "_get_neighbors() returned invalid training indices."
        )

    # Compute the true nearest indices independently.
    distances = np.linalg.norm(
        X_train - query,
        axis=1,
    )

    expected = np.argsort(distances)[:3]

    if not np.array_equal(
        neighbors,
        expected,
    ):
        _fail(
            "Incorrect nearest neighbors.\n"
            f"Expected: {expected}\n"
            f"Got: {neighbors}"
        )

    _passed()


# ============================================================
# 5. Majority vote
# ============================================================

def knn_vote_test(KNNClassifier):
    """
    Verify majority voting.
    """

    X_train = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y_train = np.array([
        0,
        0,
        1,
        0,
        1,
    ])

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    # Neighbor labels: [0, 1, 0] -> class 0
    neighbors = np.array([0, 2, 3])

    prediction = model._majority_vote(neighbors)

    if prediction != 0:
        _fail(
            f"Majority vote failed: expected 0, got {prediction}"
        )

    # Another case:
    # [1, 1, 0] -> class 1
    neighbors = np.array([2, 4, 1])

    prediction = model._majority_vote(neighbors)

    if prediction != 1:
        _fail(
            f"Majority vote failed: expected 1, got {prediction}"
        )

    _passed()


# ============================================================
# 6. Prediction
# ============================================================

def knn_predict_test(KNNClassifier):
    """
    Verify that predict() applies KNN classification correctly.
    """

    X_train = np.array([
        [-2.0, -2.0],
        [-1.0, -1.0],
        [-2.0, -1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    y_train = np.array([
        0,
        0,
        0,
        1,
        1,
        1,
    ])

    X_test = np.array([
        [-1.5, -1.5],
        [1.5, 1.5],
        [-2.0, -1.5],
        [2.0, 1.5],
    ])

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if not isinstance(predictions, np.ndarray):
        _fail(
            "predict() must return a NumPy array."
        )

    if predictions.shape != (len(X_test),):
        _fail(
            f"predict() returned shape {predictions.shape}, "
            f"expected {(len(X_test),)}."
        )

    if not np.all(
        np.isin(predictions, [0, 1])
    ):
        _fail(
            "Predictions contain unexpected class labels."
        )

    expected = np.array([0, 1, 0, 1])

    if not np.array_equal(
        predictions,
        expected,
    ):
        _fail(
            "Incorrect predictions.\n"
            f"Expected: {expected}\n"
            f"Got: {predictions}"
        )

    _passed()


# ============================================================
# 7. Score
# ============================================================

def knn_score_test(KNNClassifier):
    """
    Verify that score() returns classification accuracy.
    """

    X_train = np.array([
        [0.0],
        [1.0],
        [10.0],
        [11.0],
    ])

    y_train = np.array([
        0,
        0,
        1,
        1,
    ])

    X_test = np.array([
        [0.2],
        [0.8],
        [10.2],
        [10.8],
    ])

    y_test = np.array([
        0,
        0,
        1,
        1,
    ])

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    score = model.score(
        X_test,
        y_test,
    )

    if not np.isscalar(score):
        _fail(
            "score() must return a scalar."
        )

    if not np.isfinite(score):
        _fail(
            "score() returned NaN or infinity."
        )

    if score < 0.0 or score > 1.0:
        _fail(
            f"Accuracy must be between 0 and 1, got {score}"
        )

    predictions = model.predict(X_test)

    expected = np.mean(
        predictions == y_test
    )

    if not np.isclose(score, expected, atol=1e-8):
        _fail(
            f"score() is inconsistent with predictions. "
            f"Expected {expected}, got {score}"
        )

    _passed()


# ============================================================
# 8. Training
# ============================================================

def knn_training_test(KNNClassifier):
    """
    Verify that KNN can achieve reasonable accuracy
    on a deterministic dataset.
    """

    X_train = np.array([
        [-2.0, -2.0],
        [-1.0, -1.0],
        [-2.0, -1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    y_train = np.array([
        0,
        0,
        0,
        1,
        1,
        1,
    ])

    X_test = np.array([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    y_test = np.array([
        0,
        0,
        1,
        1,
    ])

    model = KNNClassifier(k=3)

    returned = model.fit(
        X_train,
        y_train,
    )

    if returned is not model:
        _fail(
            "fit() should return self."
        )

    predictions = model.predict(X_test)

    if predictions.shape != y_test.shape:
        _fail(
            "Prediction shape does not match y_test."
        )

    accuracy = np.mean(
        predictions == y_test
    )

    if accuracy < 0.75:
        _fail(
            f"KNN accuracy too low: "
            f"{accuracy:.3f}. Expected at least 0.75."
        )

    _passed()


# ============================================================
# 9. Prediction consistency
# ============================================================

def knn_prediction_consistency_test(KNNClassifier):
    """
    Verify consistency between predict() and the
    class's lower-level neighbor/vote operations.
    """

    X_train = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0],
    ])

    y_train = np.array([
        0,
        0,
        0,
        1,
        1,
        1,
    ])

    X_test = np.array([
        [0.2, 0.1],
        [5.2, 5.1],
        [0.8, 0.2],
        [5.8, 5.2],
    ])

    model = KNNClassifier(k=3)
    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    manual_predictions = []

    for x in X_test:

        neighbors = model._get_neighbors(x)

        prediction = model._majority_vote(
            neighbors
        )

        manual_predictions.append(prediction)

    manual_predictions = np.asarray(
        manual_predictions
    )

    if not np.array_equal(
        predictions,
        manual_predictions,
    ):
        _fail(
            "predict() is inconsistent with "
            "_get_neighbors() + _majority_vote()."
        )

    _passed()