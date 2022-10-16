import random
import numpy as np
import pandas as pd

from primo.model import PrimoClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


seed = 123
set_seed(seed)

result = pd.DataFrame()
single = pd.read_csv("PATH_TO_PROFILED_SINGLE_DATA.csv")
colo = pd.read_csv(f"PATH_TO_PROFILED_COLOCATED_DATA.csv", index_col=0)


def query_speed(trail):
    m, d, b, a = trail["model"], trail["dataset"], trail["batchsize"], trail["amp"]
    info1 = colo.query("model1 == @m and batchsize1 == @b and dataset1 == @d and amp1 == @a")
    info2 = colo.query("model2 == @m and batchsize2 == @b and dataset2 == @d and amp2 == @a")

    speed1, len1, speed2, len2 = 0, len(info1), 0, len(info2)
    if len1 > 0:
        speed1 = info1["speed1"].sum()
    if len2 > 0:
        speed2 = info2["speed2"].sum()

    avg = (speed1 + speed2) / max(len1 + len2, 1)

    return round(avg, 3)


"""Compare with original manual labeling"""
for i in range(len(single)):
    avg_speed = query_speed(single.loc[i])
    single.at[i, "avg_speed"] = avg_speed
    if avg_speed < 0.85:
        single.at[i, "auto_label"] = 2
    elif avg_speed < 0.95:
        single.at[i, "auto_label"] = 1
    else:
        single.at[i, "auto_label"] = 0


single = single.drop(columns=["dataset", "batchsize", "speed", "model"])
train_data, test_data, train_label, test_label = train_test_split(
    single.drop(columns="label"), single[["label"]], test_size=0.3, random_state=42
)

config = {"prune_factor": 0.0001}
model = PrimoClassifier(model="PrDT", model_config=config, hpo=None)
model.fit(train_data, train_label)
pred = model.predict(test_data)

acc = metrics.accuracy_score(test_label, pred)
print(f"acc: {acc:.3f}")
