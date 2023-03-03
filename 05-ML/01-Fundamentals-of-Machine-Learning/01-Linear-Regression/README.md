
# Linear Regression

## Download the dataset

The dataset is available [here](https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/ML_Houses_dataset.csv). Let's download it and store it in the `data` folder in the `01-Fundamentals-of-Machine-Learning` directory with the following commands:

```bash
cd ~/code/<user.github_nickname>/<program.challenges_repo_name>/05-ML/01-Fundamentals-of-Machine-Learning
curl https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/ML_Houses_dataset.csv > data/houses.csv
cd 01-Linear-Regression
```

## The dataset

- Each observation in the dataset represents a house
- Each column of the dataset is a feature of a house
- The target of the dataset is the sale price of the house
- You can download a detailed description of the dataset [here](https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/ML_Houses_dataset_description.txt)


## Exercise

In this exercise, you will model the relationship between the price of houses and their living surface.

Along the way, you will:

- Visualize the relationship between house surface and price
- Evaluate a Linear Regression model using K-Fold cross-validation
- Train a Linear Regression model
- Visualize the trained model
- Use the model to predict new data

## Let's go

To start the exercise, open `Linear_Regression.ipynb` in `jupyter notebook` and follow the instructions.

🚀 Your turn!
