import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

if __name__ == "__main__":
    ans_file = "data/ans.csv"
    pred_file = "data/pred.csv"
    
    ans = pd.read_csv(ans_file, usecols=["review_id", "stars"])
    pred = pd.read_csv(pred_file, usecols=["review_id", "stars"])
    
    df = pd.merge(ans, pred, how="left", on=["review_id"])
    df.fillna(0, inplace=True)
    print(len(df), sum(df["stars_y"]!=0), sum(df["stars_y"]==0))
    try:
        acc = accuracy_score(df["stars_x"], df["stars_y"])
        # acc = accuracy_score(ans["stars"], pred["stars"])
        p, r, f1, _ = precision_recall_fscore_support(
            df["stars_x"], df["stars_y"], average="macro")
            # ans["stars"], pred["stars"], average="macro")
    except:
        acc = accuracy_score(df["stars"], df["pre"])
        p, r, f1, _ = precision_recall_fscore_support(
            df["stars"], df["pre"], average="macro")
    print("accuracy:", acc, "\tprecision:", p, "\trecall:", r, "\tf1:", f1)
