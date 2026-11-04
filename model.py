"""Water potability classification and evaluation utilities."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

TARGET = "Potability"


def train_model(df: pd.DataFrame, test_size: float = 0.3, max_depth: int = 5):
    """Train and evaluate a decision tree. Returns (pipeline, metrics, report, matrix)."""
    if TARGET not in df.columns:
        raise ValueError("'Potability' 열이 필요합니다.")
    data = df.copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    if data.empty or not data[TARGET].isin([0, 1]).all():
        raise ValueError("Potability 열은 0과 1로 구성되어야 합니다.")
    y = data[TARGET].astype(int)
    X = data.drop(columns=[TARGET])
    if X.empty or X.shape[1] == 0:
        raise ValueError("학습에 사용할 독립변수 열이 필요합니다.")
    if X.select_dtypes(exclude="number").shape[1]:
        raise ValueError("독립변수는 숫자 데이터여야 합니다.")
    if X.isna().all().any():
        raise ValueError("모든 값이 결측치인 독립변수 열이 있습니다.")
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("0과 1 데이터가 각각 최소 2개 이상 필요합니다.")
    n_test = int(__import__("math").ceil(len(y) * test_size))
    if n_test < 2 or len(y) - n_test < 2:
        raise ValueError("훈련 및 테스트 데이터에 각 클래스가 포함될 만큼 충분한 데이터가 필요합니다.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=5, stratify=y
    )
    model = make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=max_depth, random_state=5))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "정확도": accuracy_score(y_test, pred),
        "정밀도": precision_score(y_test, pred, zero_division=0),
        "재현율": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
    }
    report = classification_report(y_test, pred, labels=[0, 1], target_names=["음용 부적합(0)", "음용 가능(1)"], zero_division=0)
    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    return model, metrics, report, matrix
