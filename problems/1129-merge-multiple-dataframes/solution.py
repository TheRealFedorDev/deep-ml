import pandas as pd

def solution(df1, df2, df3):
    first = pd.merge(df1, df2, on="emp_id", how='inner')
    df = pd.merge(first, df3, on="emp_id", how='left')
    return df.copy()