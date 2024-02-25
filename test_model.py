import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support


def test_model(
    df: pd.DataFrame,
    columns: list[str],
    k_value,
    sample_size=30,
    verbose=False,
    answer_column="disease",
):
    df = df.dropna()
    # Parse data trees
    X = df[columns].values
    Y = df[[answer_column]].values

    MODELS = []

    # Validator here to create
    for i in range(10):
        # Generate training data
        train_patients = df.sample(sample_size)
        trainX = train_patients[columns].values
        trainY = train_patients[[answer_column]].values

        # Build the KNN
        nn = NearestNeighbors(n_neighbors=k_value, metric="euclidean", algorithm="auto")
        fit: NearestNeighbors = nn.fit(trainX)
        MODELS.append(fit)

    # Generate test data, shortened to increase process speed
    test_patients = df.sample((len(df) - sample_size) // 2)
    testX = test_patients[columns].values
    testY = test_patients[[answer_column]].values

    # Test Predictions
    scores = []
    for model in MODELS:
        predY = []
        # Calculate NN for the test set
        distance, indices = model.kneighbors(testX)

        for i in range(len(test_patients)):
            nbrs = df.iloc[indices[i]]

            healthy = nbrs[nbrs[answer_column] == 1].count()[answer_column]
            sick = nbrs[nbrs[answer_column] == 0].count()[answer_column]

            predict = 0 if (healthy > sick) else 1

            predY.append(predict)

        score = precision_recall_fscore_support(
            testY, predY, labels=[1], zero_division=0
        )
        (p, r, f, s) = score

        if verbose:
            print(f"precision={p}, recall={r}, f-score={f}, support={s}")

        scores.append(score)

    avg_fscore = np.mean([score[2] for score in scores])
    if verbose:
        print(f"{k_value:3} | AVG FSCORE:\t{avg_fscore:3.3}")
    return avg_fscore
