import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from random import randint
from tqdm import tqdm  # Import tqdm for progress bar

i = 0
while i < 10: 
    # Load the data from CSV file
    data = pd.read_csv("S:/CHEQ/Lottery Prediction ML/STotoResult.csv")
    
    # Strip whitespace from all columns and check column names
    data.columns = data.columns.str.strip()
    expected_columns = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    if not all(col in data.columns for col in expected_columns):
        raise ValueError("Column names do not match expected columns")
    
    # Split the data into features (X) and target (y)
    X = data[['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']]
    y = data.iloc[:, 2:8]


    # Train a Random Forest Regression model
    model = RandomForestRegressor(n_estimators=1000, random_state=None)
    model.fit(X, y)

    # Generate a new set of random features for prediction
    new_data = pd.DataFrame({
        "DrawnNo1": [randint(1, 58) for _ in range(100)],
        "DrawnNo2": [randint(1, 58) for _ in range(100)],
        "DrawnNo3": [randint(1, 58) for _ in range(100)],
        "DrawnNo4": [randint(1, 58) for _ in range(100)],
        "DrawnNo5": [randint(1, 58) for _ in range(100)],
        "DrawnNo6": [randint(1, 58) for _ in range(100)],
    })

    # Ensure each set of six numbers has unique values
    for col in new_data.columns:
        new_data[col] = new_data[col].apply(lambda x: x if x in new_data[col].unique() else new_data[col].unique()[0])


    # Use tqdm to create a progress bar
    with tqdm(total=100, desc=f"Round {i+1}") as pbar:
        # Use the trained model to predict the next 6 numbers for each set of features
        predictions = model.predict(new_data)
        
        # Get the most likely set of numbers based on the predictions
        most_likely_set = predictions[0]
        for p in predictions:
            if p[0] > most_likely_set[0]:
                most_likely_set = p
        
        # Convert most_likely_set to whole numbers
        rounded_most_likely_set = [round(x) for x in most_likely_set]
        
        # Print the most likely set of numbers
        print(f"The most likely set of numbers for round {i+1} is:", rounded_most_likely_set)
        pbar.update(1)
    
    i += 1
