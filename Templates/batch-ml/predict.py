import argparse
import os
from pathlib import Path
import pandas as pd
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/models/model.pkl')
    parser.add_argument('--input', type=str, default='/data/input.csv')
    parser.add_argument('--output', type=str, default='/outputs/preds.csv')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        raise SystemExit(1)
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        raise SystemExit(1)

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)
    X = df.values
    preds = model.predict(X)
    out_df = pd.DataFrame({'prediction': preds})
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")

if __name__ == '__main__':
    main()


