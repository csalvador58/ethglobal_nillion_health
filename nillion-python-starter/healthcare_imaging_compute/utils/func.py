from matplotlib import pyplot as plt
from seaborn import histplot as sns_histplot
from pandas import DataFrame


def plot_distributions(train_data: DataFrame, num_columns: int, row_height: int, size: int, fig_name: str) -> None:
    # Plot train data in one frame
    # Calculate the number of rows and columns for subplots
    num_plots = len(train_data.columns)
    # num_cols = 7  # Number of columns for subplots
    num_rows = (num_plots - 1) // num_columns + 1  # Calculate the number of rows needed
    
    # Used to modify desired height of the frame
    # row_height = 2  # Adjust this value as needed
    fig_height = row_height * num_rows
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(size, fig_height))

    # Plot distribution of each attribute
    for i, column in enumerate(train_data.columns):
        row_index = i // num_columns
        col_index = i % num_columns
        sns_histplot(train_data[column], kde=True, ax=axes[row_index, col_index])
        axes[row_index, col_index].set_title(column)
        axes[row_index, col_index].set_xlabel('')
        axes[row_index, col_index].set_ylabel('Frequency')

    # Adjust layout
    fig.suptitle(fig_name)
    plt.tight_layout()
    plt.show()
    
    return None
    