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

# # Analysis of the data

# ## Loading and selection of the Training set

# +
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# -

dataset = pd.read_csv("drinking_water_potability.csv")
dataset.head()


train_set, test_set = train_test_split(
    dataset, train_size=0.75, random_state=42, stratify=dataset["Potability"]
)
print(train_set["Potability"].value_counts() / train_set.shape[0])
print(test_set["Potability"].value_counts() / test_set.shape[0])
del test_set

# +
feature_cols = list(filter(lambda x: x != "Potability", train_set.columns))

fig, axes = plt.subplots(2, 5, figsize=(30, 10))

for i, col in enumerate(feature_cols):
    sns.histplot(data=train_set, x=col, ax=axes[i % 2, i // 2], hue="Potability")

# +
fig, axes = plt.subplots(2, 5, figsize=(30, 10))

for i, col in enumerate(feature_cols):
    sns.boxplot(data=train_set, y=col, ax=axes[i % 2, i // 2], x="Potability")
# -

df = train_set.copy()
df["ph"] = df["ph"].round()
x = df.groupby(by="ph")["Potability"]
x = x.sum() / x.count()
del df
x.plot.bar()


# Decision: We remove waters with ph <= 1 or ph>=13 and Potability=1
# Because not really possible in real life
def process_remove_outliers_ph(df: pd.DataFrame) -> pd.DataFrame:
    """Remove waters with ph <= 1 or ph>13 and potability=1."""
    df = df[
        ~((df["Potability"] == 1) & (df["ph"].apply(lambda x: x <= 1 or x >= 13)))
    ].copy()
    return df


print(train_set.shape)
train_set = process_remove_outliers_ph(train_set)
print(train_set.shape)

# ## About the missing values

print("Pourcentages of missing values")
print(train_set.isna().sum(axis=0).sort_values(ascending=False))
(
    train_set.isna().sum(axis=0).sort_values(ascending=False) / train_set.shape[0] * 100
).round()

# +
from functools import reduce
from itertools import combinations

missing_columns = ["Sulfate", "ph", "Trihalomethanes"]
bools = {x: train_set[x].isna() for x in missing_columns}

counts = {}
options = []
for i in range(1, 4):
    for cols in combinations(missing_columns, i):
        options.append(cols)
        _ = reduce(
            lambda x, y: x & y,
            [bools[x] for x in cols]
            + [
                bools[x].apply(lambda y: y == False)
                for x in missing_columns
                if x not in cols
            ],
        )
        counts[" & ".join(cols)] = _.sum()
print("Number rows with those missing values")
pd.Series(counts).sort_values(ascending=False)
# -


# ### Observations
#
# - Only a few rows with the 3 missing values
# - Mostly sulfate missing, and mostly alone
# - ph mostly missing alone too
#

for option in options:
    sub_df = train_set[
        reduce(
            lambda x, y: x & y,
            [bools[x] for x in option]
            + [
                bools[x].apply(lambda y: y == False)
                for x in missing_columns
                if x not in option
            ],
        )
    ]

    fig, axes = plt.subplots(2, 5, figsize=(30, 10))
    plt.suptitle(
        f'Repartition of the data with the missing values: {" & ".join(option)}',
        fontsize=20,
    )
    for i, col in enumerate(sub_df.columns):
        sns.histplot(data=sub_df, x=col, ax=axes[i % 2, i // 2], hue="Potability")
