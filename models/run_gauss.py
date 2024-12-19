import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

if __name__ == "__main__":
    file_path = "data/bookcrossing/ratings.csv"
    pmf = PMF()
    pmf.set_params(
        {
            "num_feat": 50,
            "epsilon": 1,
            "_lambda": 0.1,
            "momentum": 0.8,
            "maxepoch": 15,
            "num_batches": 1000,
            "batch_size": 15000,
        }
    )
    ratings = load_rating_data(file_path, delim=",")
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = train_test_split(ratings, test_size=0.2, random_state=12)
    pmf.fit(train, test)

    # sample users who were in the test set
    # test_users = np.unique(test[:, 0])
    rng = np.random.default_rng(12)
    test_users = rng.choice(np.unique(test[:, 0]), size=1200, replace=False)
    # predict ratings for the sample of users
    predictions = []
    for user in test_users:
        user_predictions = pmf.predict(user)
        predictions.append(user_predictions)
    predictions = np.array(predictions)
    # save the predictions
    np.save("predictions/gauss_predictions_bc_50.npy", predictions)
    np.save("predictions/gauss_test_users_bc_50.npy", test_users)
