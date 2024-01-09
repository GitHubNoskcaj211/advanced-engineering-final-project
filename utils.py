import torch
import pandas as pd
import matplotlib.pyplot as plt

def dataframe_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

def tensor_to_dataframe(tensor, similar_df):
    return pd.DataFrame(tensor.numpy(), columns=similar_df.columns)

def clean_dataframe(dataframe):
    return dataframe.loc[:, ~dataframe.columns.isin(['positive', 'contract'])]

def plot_losses(title: str, losses):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')    
    return fig

def combine_dataframes(df1, df2, N, percent_from_dataframe_1):
    if percent_from_dataframe_1 < 0 or percent_from_dataframe_1 > 1:
        raise ValueError("Percentage must be between 0 and 1")
    if N is None:
        # As much as possible to keep the percentage
        if (df1.shape[0] + df2.shape[0]) * percent_from_dataframe_1 > df1.shape[0]:
            rows_from_df1 = int(df1.shape[0])
            rows_from_df2 = int(rows_from_df1 / percent_from_dataframe_1 - rows_from_df1)
        else:
            rows_from_df2 = int(df2.shape[0])
            rows_from_df1 = int(rows_from_df2 / (1 - percent_from_dataframe_1) - rows_from_df2)
    else:
        rows_from_df1 = int(N * percent_from_dataframe_1)
        rows_from_df2 = N - rows_from_df1
    
    rows_from_df1 = min(rows_from_df1, df1.shape[0])
    rows_from_df2 = min(rows_from_df2, df2.shape[0])

    sampled_df1 = df1.sample(n=rows_from_df1)
    sampled_df2 = df2.sample(n=rows_from_df2)

    combined_df = pd.concat([sampled_df1, sampled_df2], ignore_index=True)

    return combined_df