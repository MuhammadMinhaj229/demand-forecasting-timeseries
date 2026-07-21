import pandas as pd

def main():
    print("Loading data...")
    df = pd.read_csv('data/bike_sharing.csv')

    print("\n--- Data Overview ---")
    print(df.head())

    print("\n--- Data Info ---")
    print(df.info())

    print("\n--- Summary Statistics ---")
    print(df.describe())

    print("\n--- Checking for Missing Values ---")
    print(df.isnull().sum())

    print("\n--- EDA complete ---")

if __name__ == "__main__":
    main()
