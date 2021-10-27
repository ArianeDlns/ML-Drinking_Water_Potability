# +
# Set full width on jupyter notebook navigator
from IPython.core.display import HTML, display

display(HTML("<style>.container { width:100% !important; }</style>"))

# +
# For dark mode only, font in white
import matplotlib as mpl

mpl.rcParams.update(
    {"axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"}
)
# -

# # Pipeline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dataset = pd.read_csv("../drinking_water_potability.csv")
dataset.head()

# +
train_set, test_set = train_test_split(
    dataset, train_size=0.75, random_state=42, stratify=dataset["Potability"]
)

target = "Potability"
feature_names = list(filter(lambda x: x != target, train_set.columns))

print(f"Elements in train: {train_set.shape[0]}")
print(f"Elements in test : {test_set.shape[0]}")
print(train_set["Potability"].value_counts() / train_set.shape[0])
print(test_set["Potability"].value_counts() / test_set.shape[0])
# -

# ## Removing the outliers before starting

# +
from utils import process_remove_outliers_ph

processors = [process_remove_outliers_ph]

# +
print("=" * 10, "TRAIN", "=" * 10)
print(f"Initial size: {train_set.shape}")
for processor in processors:
    train_set = processor(train_set)
    print(f"Apply {processor.__name__}, new size: {train_set.shape}")

print("\n" * 2, "=" * 10, "TEST", "=" * 10)
print(f"Initial size: {test_set.shape}")
for processor in processors:
    test_set = processor(test_set)
    print(f"Apply {processor.__name__}, new size: {test_set.shape}")
# -

# ## All the transformers and steps of the pipeline


# ### Definition of the transformer pipeline

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# +
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

transformer = Pipeline(
    [
        ("Missing_values_handler", IterativeImputer(max_iter=10, random_state=0)),
        ("Scaler", StandardScaler()),
    ]
)
transformer
# -

# ## Models exploration

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# +
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = [
    DecisionTreeClassifier(max_depth=7),
    LogisticRegression(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    RandomForestClassifier(),
    ExtraTreesClassifier(n_estimators=10),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
]

metrics = [accuracy_score, f1_score, precision_score, recall_score]
results = []

X_train, X_val, y_train, y_val = train_test_split(
    train_set[feature_names], train_set[target], train_size=0.8, random_state=20
)
for model in tqdm(models, desc="Training different models", unit="model"):
    scores = {"name": model.__class__.__name__}
    model = Pipeline([("transformer", transformer), ("model", model)])
    scores = {
        **scores,
        **{
            "cv_"
            + func.__name__: np.mean(
                cross_val_score(
                    model, X_train, y_train, cv=5, scoring=make_scorer(func)
                )
            )
            for func in metrics
        },
    }
    results.append(scores)
results_df = pd.DataFrame.from_records(results)
# -

results_df.sort_values(by=["cv_precision_score"], ascending=False)

results_df.set_index("name")["cv_precision_score"].sort_values().plot.bar(
    ylim=[0, 1], grid=True
)

X_train, y_train = train_set[feature_names], train_set[target]
X_test, y_test = test_set[feature_names], test_set[target]

# ## Training

from sklearn.model_selection import GridSearchCV

# ###  Grid search area

# ### SVC

# +
# Train using a grid search to find the best params
# For a SVC

param_grid = {
    "C": [0.1, 10, 50, 100],
    "kernel": ["linear", "poly", "sigmoid", "rbf"],
    "class_weight": [None, "balanced"],
}
param_grid = {"model__" + k: v for k, v in param_grid.items()}
pipeline_gs = Pipeline([("transformer", transformer), ("model", SVC())])
print(pipeline_gs.get_params())
grid_search = GridSearchCV(
    pipeline_gs,
    param_grid=param_grid,
    cv=5,
    verbose=10,
    n_jobs=-1,
    scoring=["f1_weighted", "accuracy", "average_precision"],
    refit=False,
)
grid_search.fit(X_train, y_train)
# -

_ = pd.DataFrame(grid_search.cv_results_)
_ = _.sort_values(by=["mean_test_average_precision"], ascending=False).reset_index(
    drop=True
)
_.head()

print(_.loc[1])
print(_.at[1, "params"])

print(
    """For the SVC, after Grid Search, we have the following:
mean_fit_time                                                             0.303844
std_fit_time                                                              0.021136
mean_score_time                                                           0.089625
std_score_time                                                             0.00434
param_model__C                                                                  10
param_model__class_weight                                                     None
param_model__kernel                                                            rbf
params                           {'model__C': 10, 'model__class_weight': None, ...
split0_test_f1_weighted                                                    0.69426
split1_test_f1_weighted                                                   0.626988
split2_test_f1_weighted                                                   0.647527
split3_test_f1_weighted                                                   0.635514
split4_test_f1_weighted                                                   0.643503
mean_test_f1_weighted                                                     0.649558
std_test_f1_weighted                                                      0.023434
rank_test_f1_weighted                                                            1
split0_test_accuracy                                                      0.700611
split1_test_accuracy                                                      0.643585
split2_test_accuracy                                                      0.663951
split3_test_accuracy                                                      0.651731
split4_test_accuracy                                                      0.659878
mean_test_accuracy                                                        0.663951
std_test_accuracy                                                          0.01962
rank_test_accuracy                                                               1
split0_test_average_precision                                             0.622592
split1_test_average_precision                                             0.579198
split2_test_average_precision                                             0.587247
split3_test_average_precision                                             0.571751
split4_test_average_precision                                             0.591173
mean_test_average_precision                                               0.590392
std_test_average_precision                                                0.017436
rank_test_average_precision                                                      2
Name: 1, dtype: object
{'model__C': 10, 'model__class_weight': None, 'model__kernel': 'rbf'}"""
)

# ### Random Forest

# +
# Train using a grid search to find the best params
# For a random forest

param_grid = {
    "n_estimators": [10, 50, 100, 150, 200, 250, 300],
    "max_features": list(range(2, len(X_train.columns) + 1)),
}
param_grid = {"model__" + k: v for k, v in param_grid.items()}
pipeline_gs = Pipeline(
    [("transformer", transformer), ("model", RandomForestClassifier())]
)
print(pipeline_gs.get_params())

grid_search = GridSearchCV(
    pipeline_gs,
    param_grid=param_grid,
    cv=5,
    verbose=10,
    n_jobs=-1,
    scoring=["f1_weighted", "accuracy", "average_precision"],
    refit=False,
)
grid_search.fit(X_train, y_train)
# -

_ = pd.DataFrame(grid_search.cv_results_)
print(_.columns)
_ = _.sort_values(by=["mean_test_average_precision"], ascending=False).reset_index(
    drop=True
)
_.head()

IDX = 0
print(_.loc[IDX])
print(_.at[IDX, "params"])

print(
    """mean_fit_time                                                             2.428444
std_fit_time                                                              0.049107
mean_score_time                                                           0.080467
std_score_time                                                            0.002905
param_model__max_features                                                        7
param_model__n_estimators                                                      250
params                           {'model__max_features': 7, 'model__n_estimator...
split0_test_f1_weighted                                                   0.672308
split1_test_f1_weighted                                                   0.673643
split2_test_f1_weighted                                                   0.646175
split3_test_f1_weighted                                                   0.611159
split4_test_f1_weighted                                                     0.6499
mean_test_f1_weighted                                                     0.650637
std_test_f1_weighted                                                      0.022706
rank_test_f1_weighted                                                            1
split0_test_accuracy                                                      0.687023
split1_test_accuracy                                                      0.694656
split2_test_accuracy                                                      0.671756
split3_test_accuracy                                                      0.653944
split4_test_accuracy                                                      0.673469
mean_test_accuracy                                                         0.67617
std_test_accuracy                                                         0.014002
rank_test_accuracy                                                               2
split0_test_average_precision                                             0.621502
split1_test_average_precision                                             0.576277
split2_test_average_precision                                             0.589649
split3_test_average_precision                                             0.592778
split4_test_average_precision                                             0.596943
mean_test_average_precision                                                0.59543
std_test_average_precision                                                 0.01476
rank_test_average_precision                                                      1
Name: 0, dtype: object
{'model__max_features': 7, 'model__n_estimators': 250}"""
)

# +
# Train using a grid search to find the best params
# For a gradientboosting model

param_grid = {
    "n_estimators": [10, 50, 100, 150, 200, 250, 300],
    "learning_rate": [1.0, 0.5, 5],
    "loss": ["deviance", "exponential"],
}
param_grid = {"model__" + k: v for k, v in param_grid.items()}
pipeline_gs = Pipeline(
    [("transformer", transformer), ("model", GradientBoostingClassifier())]
)
print(pipeline_gs.get_params())

grid_search = GridSearchCV(
    pipeline_gs,
    param_grid=param_grid,
    cv=5,
    verbose=10,
    n_jobs=-1,
    scoring=["f1_weighted", "accuracy", "average_precision"],
    refit=False,
)
grid_search.fit(X_train, y_train)
# -

_ = pd.DataFrame(grid_search.cv_results_)
print(_.columns)
_ = _.sort_values(by=["mean_test_average_precision"], ascending=False).reset_index(
    drop=True
)
_.head()

IDX = 0
print(_.loc[IDX])
print(_.at[IDX, "params"])


# ### Final model training

from sklearn.ensemble import VotingClassifier

model = GradientBoostingClassifier(learning_rate=0.5, n_estimators=250)
model = SVC()
model = VotingClassifier(
    estimators=[
        ("svc", SVC()),
        ("gbc", GradientBoostingClassifier(learning_rate=0.5, n_estimators=100)),
        ("Random_forest", RandomForestClassifier(n_estimators=100, max_features=7)),
    ],
    voting="hard",
)
pipeline = Pipeline([("transformer", transformer), ("model", model)], verbose=True)

pipeline.fit(X_train, y_train)

# ### Check on training set

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

# +
y_train_pred = pipeline.predict(X_train)

print(classification_report(y_train, y_train_pred))

ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred)).plot()
# -

# ## Check on the test set

# +
y_test_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_test_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred)).plot()
