import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def bars(df, col):
    col_df = pd.DataFrame(df.groupby([col, 'Default']).size().unstack())
    col_df.plot(kind='bar', stacked = True)
    
def hists(df, col):
    df[col].hist()