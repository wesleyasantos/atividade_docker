import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/data/iris.csv')
    parser.add_argument('--out', type=str, default='/models/model.pkl')
    parser.add_argument('--metrics', type=str, default='/models/metrics.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    if os.path.exists(args.data):
        df = pd.read_csv(args.data)
        X = df.drop(columns=df.columns[-1]).values
        y = df.iloc[:, -1].values
    else:
        iris = load_iris(as_frame=True)
        X = iris.data.values
        y = iris.target.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    # Persist
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)

    Path(os.path.dirname(args.metrics)).mkdir(parents=True, exist_ok=True)
    with open(args.metrics, 'w', encoding='utf-8') as f:
        json.dump({"accuracy": acc}, f, indent=2)

    print(f"Model saved to {args.out}")
    print(f"Metrics saved to {args.metrics}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()


