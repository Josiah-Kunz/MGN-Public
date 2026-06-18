import os
from sklearn.ensemble import GradientBoostingRegressor
from meshgraphnet import MLObject

def main():

    # The data we want to train on
    data_file = os.path.join("cantilever", "results", "cantilever_fem_results.csv")

    # Construct the machine learning object
    ml = MLObject(
        data=data_file,
        features=["x (m)", "y (m)"],
        objectives=["von_mises (GPa)"],
        name="Cantilever Analysis"
    )

    # Train + test machine learning
    ml.train(model=GradientBoostingRegressor()) # Default is sklearn linear regression model, which isn't great for cantilevers
    ml.evaluate_on_unseen_data()
    ml.plot_predictions()

if __name__ == "__main__":
    main()