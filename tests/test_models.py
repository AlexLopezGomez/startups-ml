import pandas as pd
import numpy as np
import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src.models.train_model import train_model
from src.models.predict_model import load_model, predict
from src.models.evaluate_model import evaluate_model
from src.models.bias_variance import perform_bias_variance_analysis


def sample_data():
    X = pd.DataFrame({'x': [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    return X, y


def test_train_model():
    X, y = sample_data()
    model = DecisionTreeClassifier(max_depth=1, random_state=42)
    trained = train_model(X, y, model)
    assert trained is model
    # Debe poder predecir
    preds = trained.predict(X)
    assert len(preds) == len(y)


def test_load_and_predict(tmp_path):
    X, y = sample_data()
    dummy = DummyClassifier(strategy='constant', constant=1)
    dummy.fit(X, y)
    path = tmp_path / 'dummy.pkl'
    joblib.dump(dummy, path)
    loaded = load_model(path)
    preds = predict(loaded, X)
    assert isinstance(preds, np.ndarray)
    assert preds.tolist() == [1] * len(y)


def test_evaluate_model():
    X, y = sample_data()
    dummy = DummyClassifier(strategy='constant', constant=1)
    dummy.fit(X, y)
    acc, report, cm = evaluate_model(dummy, X, y)
    assert isinstance(acc, float)
    assert isinstance(report, dict)
    assert cm.shape == (2, 2)
    # accuracy debe ser entre 0 y 1
    assert 0.0 <= acc <= 1.0


def test_perform_bias_variance_analysis():
    X, y = sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    params, train_scores, test_scores, best = perform_bias_variance_analysis(
        X_train, X_test, y_train, y_test, param_range=[1]
    )
    assert params == [1]
    assert isinstance(train_scores, list)
    assert isinstance(test_scores, list)
    assert best == 1


def test_load_model_not_found(tmp_path):
    """Verifica que load_model lance FileNotFoundError cuando el archivo no existe."""
    nonexistent = tmp_path / "no_model.pkl"
    with pytest.raises(FileNotFoundError):
        load_model(nonexistent)
