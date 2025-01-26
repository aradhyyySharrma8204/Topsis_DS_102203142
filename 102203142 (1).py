
import sys
import pandas as pd
import numpy as np

def topsis(input_file, weights, impacts, result_file):
    try:
        # Read the input file
        data = pd.read_csv(input_file)

        # Input validation
        if len(data.columns) < 3:
            raise ValueError("Input file must have at least 3 columns")

        if len(weights) != (len(data.columns) - 1) or len(impacts) != (len(data.columns) - 1):
            raise ValueError("Number of weights and impacts must match the number of columns (excluding the first column).")

        if not all(impact in ['+', '-'] for impact in impacts):
            raise ValueError("Impacts must be either '+' or '-'")

        # Convert data (2nd to last columns) to numeric
        for col in data.columns[1:]:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isnull().any():
                raise ValueError(f"Column {col} contains non-numeric values.")

        # Normalize the data
        norm_data = data.iloc[:, 1:].div(np.sqrt((data.iloc[:, 1:]**2).sum()), axis=1)

        # Apply weights
        norm_data = norm_data * weights

        # Determine ideal best and worst
        ideal_best = []
        ideal_worst = []
        for i in range(len(impacts)):
            if impacts[i] == '+':
                ideal_best.append(norm_data.iloc[:, i].max())
                ideal_worst.append(norm_data.iloc[:, i].min())
            else:
                ideal_best.append(norm_data.iloc[:, i].min())
                ideal_worst.append(norm_data.iloc[:, i].max())

        # Calculate distances
        distances_best = np.sqrt(((norm_data - ideal_best)**2).sum(axis=1))
        distances_worst = np.sqrt(((norm_data - ideal_worst)**2).sum(axis=1))

        # Calculate Topsis Score
        scores = distances_worst / (distances_best + distances_worst)
        data['Topsis Score'] = scores

        # Rank the alternatives
        data['Rank'] = data['Topsis Score'].rank(ascending=False).astype(int)

        # Save the result
        data.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
    else:
        input_file = sys.argv[1]
        weights = [float(w) for w in sys.argv[2].split(',')]
        impacts = sys.argv[3].split(',')
        result_file = sys.argv[4]
        topsis(input_file, weights, impacts, result_file)
