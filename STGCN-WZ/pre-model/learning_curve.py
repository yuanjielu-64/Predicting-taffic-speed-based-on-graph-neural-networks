import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn import model_selection
from sklearn.metrics.scorer import make_scorer
from math import sqrt
from sklearn.metrics import mean_squared_error

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes= np.linspace(0.1, 1, 10)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes= train_sizes, return_times = True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")

    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    print(title)
    return plt

fig, axes = plt.subplots(3, 2, figsize=(10, 15))

df = pd.read_csv(r'Y:\Project\tmc_final.csv')
#df = pd.read_csv('/Users/yuanjielu/Desktop/Research/Project/tmc_final.csv', index_col = 0)

def rmse(true, predicted):
  return sqrt(mean_squared_error(np.expm1(true), np.expm1(predicted)))

target = df.speed
target = np.array(target)
useless = ['average_speed','reference_speed','travel_time_minutes','Date','measurement_tstamp',
            'total cost','person_hrs_of_delay','Vehicle_hrs_of_delay', 'speed','total VMT'
           ]
dataset = df.drop(columns = useless)

enc = preprocessing.LabelEncoder()
dataset.tmc_code = enc.fit_transform(dataset.tmc_code)
dataset.direction = enc.fit_transform(dataset.direction)
dataset.Facts = enc.fit_transform(dataset.Facts)
#dataset.to_csv("day_.csv")
y = np.log1p(target)

x_train, x_test, y_train, y_test = model_selection. train_test_split(dataset, y, test_size=0.25, random_state = 1234)
my_scorer = make_scorer(rmse, greater_is_better=True)

title = "Learning Curves (KNN)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = KNeighborsRegressor(weights='distance', n_neighbors= 10)
plot_learning_curve(estimator, title, x_train, y_train, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=2)
plt.show()

title = r"Learning Curves (SVR)"
estimator = SVR(kernel='rbf', C= 0.1, gamma = 1 , epsilon = 0.1 )
plot_learning_curve(estimator, title, x_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=2)
plt.show()

title = r"Learning Curves (RF)"
estimator = RandomForestRegressor(random_state=1234,n_estimators = 50, max_depth= 16, bootstrap = True, max_features= 'auto')
plot_learning_curve(estimator, title, x_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=2)

plt.show()