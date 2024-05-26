from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

def compare_model_performance(vector_1, vector_2, description):
    vector_1 = np.array(vector_1)
    vector_2 = np.array(vector_2)

    mae = mean_absolute_error(vector_1, vector_2)
    euclidean = np.linalg.norm(vector_1 - vector_2)
    spearman_corr, _ = spearmanr(vector_1, vector_2)

    return pd.Series({
        'Comparison': description,
        'MAE': mae,
        'Euclidean Distance': euclidean,
        'Spearman Correlation': spearman_corr
    })



#comparison_results_df = pd.DataFrame([compare_model_performance(v1, v2, desc) for v1, v2, desc in model_pairs])

# Redefining the function after environment reset
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_distributions(recom, clust, knn, recom_norm, clust_norm, knn_norm):
    """
    Plots histograms and boxplots for hit scores and normalized hit scores for three models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  

    # Histograms for hit scores
    sns.histplot(recom, ax=axes[0, 0], color="blue", label="Recom", kde=True, alpha=0.5)
    sns.histplot(clust, ax=axes[0, 1], color="red", label="Clust", kde=True, alpha=0.5)
    sns.histplot(knn, ax=axes[0, 2], color="green", label="KNN", kde=True, alpha=0.5)
    axes[0, 0].set_title('Distribution of Recom Hit Scores')
    axes[0, 1].set_title('Distribution of Clust Hit Scores')
    axes[0, 2].set_title('Distribution of KNN Hit Scores')
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[0, 2].legend()

    # Histograms for normalized hit scores
    sns.histplot(recom_norm, ax=axes[1, 0], color="blue", label="Recom Norm", kde=True, alpha=0.5)
    sns.histplot(clust_norm, ax=axes[1, 1], color="red", label="Clust Norm", kde=True, alpha=0.5)
    sns.histplot(knn_norm, ax=axes[1, 2], color="green", label="KNN Norm", kde=True, alpha=0.5)
    axes[1, 0].set_title('Distribution of Normalized Recom Hit Scores')
    axes[1, 1].set_title('Distribution of Normalized Clust Hit Scores')
    axes[1, 2].set_title('Distribution of Normalized KNN Hit Scores')
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

# Call the function with placeholder data
#plot_model_distributions(recom, clust, knn, recom_norm, clust_norm, knn_norm)
    

import seaborn as sns
import matplotlib.pyplot as plt

def plot_model_histograms_sns(data_series):
    # Number of rows and cols for the subplot grid, assuming a square layout for simplicity
    num_items = len(data_series)
    grid_size = int(num_items ** 0.5)
    if grid_size ** 2 < num_items:
        grid_size += 1  # Ensure enough space for all items

    # Create figure and axes for subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the array for easy iteration

    # Add histograms to the subplots
    for ax, (name, data) in zip(axs, data_series.items()):
        sns.histplot(data, kde=True, ax=ax)
        ax.set_title(name)

    # Hide any unused subplots
    for i in range(len(data_series), len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout()
    return fig


    
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_boxplots(recom, clust, knn, recom_norm, clust_norm, knn_norm):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Boxplot for Hit Scores
    sns.boxplot(data=[recom, clust, knn], ax=axes[0])
    axes[0].set_xticklabels(['Recom', 'Clust', 'KNN'])
    axes[0].set_title('Box Plot of Hit Scores')
    axes[0].set_ylabel('Hit Scores')

    # Boxplot for Normalized Hit Scores
    sns.boxplot(data=[recom_norm, clust_norm, knn_norm], ax=axes[1])
    axes[1].set_xticklabels(['Recom Norm', 'Clust Norm', 'KNN Norm'])
    axes[1].set_title('Box Plot of Normalized Hit Scores')
    axes[1].set_ylabel('Normalized Hit Scores')

    plt.tight_layout()
    plt.show()

# Call the function with placeholder data
#plot_model_boxplots(recom, clust, knn, recom_norm, clust_norm, knn_norm)
    
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_model_boxplots_plotly(recom, clust, knn, recom_norm, clust_norm, knn_norm):
    # Create subplots: one row, two cols
    fig = sp.make_subplots(rows=2, cols=1, subplot_titles=('Box Plot of Hit Scores', 'Box Plot of Normalized Hit Scores'))

    # Boxplot for Hit Scores
    fig.add_trace(go.Box(y=recom, name='Recom'), row=1, col=1)
    fig.add_trace(go.Box(y=clust, name='Clust'), row=1, col=1)
    fig.add_trace(go.Box(y=knn, name='KNN'), row=1, col=1)

    # Boxplot for Normalized Hit Scores
    fig.add_trace(go.Box(y=recom_norm, name='Recom Norm'), row=2, col=1)
    fig.add_trace(go.Box(y=clust_norm, name='Clust Norm'), row=2, col=1)
    fig.add_trace(go.Box(y=knn_norm, name='KNN Norm'), row=2, col=1)

    # Update layout for a clean look
    fig.update_layout(height=800, title_text="Box Plots of Hit Scores and Normalized Hit Scores")
    fig.show()

# Assuming recom, clust, knn, recom_norm, clust_norm, and knn_norm are defined, call the function
#plot_model_boxplots_plotly(recom, clust, knn, recom_norm, clust_norm, knn_norm)
    

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_model_boxplots_plotly(data_series):
    # Number of rows and cols for the subplot grid, assuming a square layout for simplicity
    num_items = len(data_series)
    grid_size = int(num_items ** 0.5)
    if grid_size ** 2 < num_items:
        grid_size += 1  # Ensure enough space for all items

    # Create subplots
    fig = make_subplots(rows=grid_size, cols=grid_size, subplot_titles=list(data_series.keys()))

    # Add box plots to the subplots
    row_col_index = 1  # Start from the first subplot
    for name, data in data_series.items():
        row = (row_col_index - 1) // grid_size + 1
        col = (row_col_index - 1) % grid_size + 1
        fig.add_trace(go.Box(y=data, name=name), row=row, col=col)
        row_col_index += 1

    # Update layout for a clean look
    fig.update_layout(height=600, width=1000, title_text="Box Plots of Different Data Series")
    return fig



import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_model_comparison_boxplots(data_series, subplot_titles=("Box Plot of Models")):
    # Number of series
    n_series = len(data_series)

    # Create subplots: one row, n_series cols
    fig = make_subplots(rows=1, cols=n_series, subplot_titles=subplot_titles)

    # Add each data series as a separate box plot
    for i, (series_name, series_data) in enumerate(data_series.items(), start=1):
        fig.add_trace(go.Box(y=series_data, name=series_name), row=1, col=i)

    # Update layout for a clean look
    fig.update_layout(height=600, title_text="Model Comparison Box Plots")
    return fig


import numpy as np
import pandas as pd
from scipy.stats import shapiro


def test_normality(*args):
    # Initialize an empty list to hold the results
    results = []
    
    # Loop through each argument (series) provided to the function
    for idx, data in enumerate(args):
        # Perform Shapiro-Wilk test
        stat, p = shapiro(data)
        
        # Determine if the distribution can be considered Gaussian
        alpha = 0.05
        normality = 'Gaussian' if p > alpha else 'Not Gaussian'
        
        # Append the results to the list
        results.append({
            'Model': f'Model {idx + 1}',
            'Statistics': stat,
            'P-Value': p,
            'Normality': normality
        })
    
    # Convert the results list to a DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return results_df

# Assuming recom, clust, knn, recom_norm, clust_norm, and knn_norm are defined as NumPy arrays or pandas Series
# Call the function with your data series
#results_df = test_normality(recom, clust, knn, recom_norm, clust_norm, knn_norm)

from scipy.stats import kstest, norm

def test_normality_kolmogorov(data_series):
    results = []
    
    # Loop through each data series
    for model, data in data_series.items():
        # Perform Kolmogorov-Smirnov test against normal distribution
        stat, p = kstest(data, 'norm')
        
        # Determine if the distribution can be considered Gaussian
        alpha = 0.05
        normality = 'Gaussian' if p > alpha else 'Not Gaussian'
        
        # Append the results to the list
        results.append({
            'Model': model,
            'Statistics': stat,
            'P-Value': p,
            'Normality': normality
        })
    
    # Convert the results list to a DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return results_df


import numpy as np
import pandas as pd
from scipy.stats import iqr

def compare_model_variability(vectors, vector_names):
    """
    Calculate variability measures for given vectors and assign appropriate names.

    :param vectors: List of vectors for which to calculate variability measures.
    :param vector_names: Corresponding names for the vectors.
    :return: DataFrame with calculated metrics.
    """
    metrics_list = []
    for vector, name in zip(vectors, vector_names):
        vector_np = np.array(vector)
        variance = np.var(vector_np)
        std_dev = np.std(vector_np)
        cv = std_dev / np.mean(vector_np) if np.mean(vector_np) != 0 else float('inf')
        iqr_value = iqr(vector_np)
        
        metrics_list.append({
            "Model": name,
            "Variance": variance,
            "Standard Deviation": std_dev,
            "Coefficient of Variation": cv,
            "Interquartile Range": iqr_value
        })
    
    return pd.DataFrame(metrics_list)

from scipy.stats import kendalltau
import pandas as pd

def calculate_rbo(list1, list2, p=0.9):
    """
    Calculate Rank-Biased Overlap (RBO) between two lists.
    p: The persistence probability to continue.
    """
    if not list1 or not list2:
        return 0.0

    sl, ll = sorted([(len(list1), list1), (len(list2), list2)], key=lambda x: x[0])
    s, S = sl[1], ll[1]  # shorter list, longer list
    s_len, l_len = len(s), len(S)
    if s_len == 0: return 0.0

    weighted_overlap = sum([(len(set(s[:i]) & set(S[:i])) / i) * pow(p, i) for i in range(1, s_len + 1)])
    correction = p**s_len * (len(set(s) & set(S[s_len:])) / s_len - (weighted_overlap / pow(1 - p, 2)))
    rbo = ((1 - p) / p) * (weighted_overlap - correction)
    return rbo

def compare_model_scores(scores, model_names):
    results = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            list1 = scores[i].tolist() if hasattr(scores[i], 'tolist') else list(scores[i])
            list2 = scores[j].tolist() if hasattr(scores[j], 'tolist') else list(scores[j])
            rbo_score = calculate_rbo(list1, list2)
            tau, p_value = kendalltau(list1, list2)
            results.append({
                "Model Pair": f"{model_names[i]} vs {model_names[j]}",
                "RBO Score": rbo_score,
                "Kendall’s Tau": tau,
                "P-value": p_value
            })
    return pd.DataFrame(results)


import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def perform_t_tests(*args, model_names):

    # Ensure there are at least two series to compare
    if len(args) < 2:
        raise ValueError("At least two series must be provided for comparison.")
    
    if len(args) != len(model_names):
        raise ValueError("Number of model names must match the number of series provided.")
    
    # Initialize a list to hold comparison results
    comparison_results = []
    
    # Generate all unique pairs for comparison
    for i in range(len(args)):
        for j in range(i+1, len(args)):
            vector_1, vector_2 = args[i], args[j]
            model_1, model_2 = model_names[i], model_names[j]
            t_stat, p_value = ttest_rel(vector_1, vector_2)
            
            # Append comparison results
            comparison_results.append({
                'Comparison': f'{model_1} vs. {model_2}',
                'Average Hits Model 1': np.mean(vector_1),
                'Average Hits Model 2': np.mean(vector_2),
                'T-statistic': t_stat,
                'P-value': p_value,
                'Significance': "Significant" if p_value < 0.05 else "Not Significant"
            })
    
    # Convert comparison results to DataFrame for easy viewing
    comparison_df = pd.DataFrame(comparison_results)
    return comparison_df

# Example 
# model_names = ['Recom Norm', 'Clust Norm', 'KNN Norm']
# comparison_df = perform_t_tests(recom_norm, clust_norm, knn_norm, model_names=model_names)
# print(comparison_df)


from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd

def perform_mann_whitney_tests(*args, model_names):
    """
    Perform Mann-Whitney U tests for all unique pairs of provided datasets.
    
    Parameters:
    - args: Variable length argument list, each argument being a dataset (numpy array or list).
    - model_names: List of names corresponding to each dataset in args.
    
    Returns:
    - DataFrame with columns for comparison pairs, U statistics, p-values, and significance.
    """
    # Ensure there are at least two series to compare and matching names
    if len(args) < 2:
        raise ValueError("At least two series must be provided for comparison.")
    if len(args) != len(model_names):
        raise ValueError("Number of model names must match the number of series provided.")
    
    # Initialize list to store test results
    test_results = []
    
    # Perform Mann-Whitney U test for each pair of datasets
    for i in range(len(args)):
        for j in range(i+1, len(args)):
            data1, data2 = args[i], args[j]
            name1, name2 = model_names[i], model_names[j]
            
            # Perform Mann-Whitney U test
            u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Determine significance
            significance = 'Significant' if p_value < 0.01 else 'Not Significant'
            
            # Append results to the list
            test_results.append({
                'Comparison': f'{name1} vs. {name2}',
                'U-statistic': u_stat,
                'P-value': p_value,
                'Significance': significance
            })
    
    # Convert list of results to DataFrame
    results_df = pd.DataFrame(test_results)
    return results_df

import umap
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

def analyze_embeddings(embeddings, n_neighbors=25, min_dist=0.01, n_components=3, num_clusters=5, random_state=42):
    """
    Analyzes and visualizes semantic embeddings using UMAP for dimensionality reduction and KMeans for clustering.

    Parameters:
    - embeddings: The high-dimensional embeddings array.
    - n_neighbors, min_dist, n_components: UMAP parameters for dimensionality reduction.
    - num_clusters: Number of clusters for KMeans.
    - random_state: Seed for reproducibility.

    Returns:
    - A Plotly figure object visualizing the embeddings in reduced dimensions, colored by cluster labels.
    """
    # Dimensionality Reduction with UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    umap_embeddings = reducer.fit_transform(embeddings)

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    # Visualization with Plotly
    spectral_scale = [
        "#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b",
        "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"
    ]
    fig = px.scatter(
        x=umap_embeddings[:, 0], y=umap_embeddings[:, 1],
        color=cluster_labels,
        title="UMAP Projection of Semantic Embeddings",
        color_continuous_scale=spectral_scale,
        labels={"color": "Cluster Label"}
    )
    fig.update_layout(
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        coloraxis_colorbar=dict(title='Cluster Label'),
        template='plotly_dark',
        height=800,
        width=1000
    )
    
    return fig


def analyze_embeddings_3D(embeddings, n_neighbors=25, min_dist=0.01, n_components=3, num_clusters=5, random_state=42):
    """
    Analyzes and visualizes semantic embeddings using UMAP for dimensionality reduction and KMeans for clustering.

    Parameters:
    - embeddings: The high-dimensional embeddings array.
    - n_neighbors, min_dist, n_components: UMAP parameters for dimensionality reduction.
    - num_clusters: Number of clusters for KMeans.
    - random_state: Seed for reproducibility.

    Returns:
    - A Plotly figure object visualizing the embeddings in reduced dimensions, colored by cluster labels.
    """
    # Dimensionality Reduction with UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    umap_embeddings = reducer.fit_transform(embeddings)

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    # Visualization with Plotly
    spectral_scale = [
        "#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b",
        "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"
    ]
    
    # Create the 3D scatter plot
    fig = px.scatter_3d(
        x=umap_embeddings[:, 0],
        y=umap_embeddings[:, 1],
        z=umap_embeddings[:, 2],
        color=cluster_labels,
        title="3D UMAP Projection of Semantic Embeddings",
        color_continuous_scale=spectral_scale,
        labels={"color": "Cluster Label"}
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3"
        ),
        coloraxis_colorbar=dict(title='Cluster Label'),
        template='plotly_dark',
        height=800,
        width=1000
    )
    
    return fig


# def analyze_embeddings_mark(df, embeddings, n_neighbors=25, min_dist=0.01, n_components=3, num_clusters=5, random_state=42):
#     """
#     Analyzes and visualizes semantic embeddings using UMAP for dimensionality reduction and KMeans for clustering.

#     Parameters:
#     - df: DataFrame containing 'label' column.
#     - embeddings: The high-dimensional embeddings array.
#     - n_neighbors, min_dist, n_components: UMAP parameters for dimensionality reduction.
#     - num_clusters: Number of clusters for KMeans.
#     - random_state: Seed for reproducibility.

#     Returns:
#     - A Plotly figure object visualizing the embeddings in reduced dimensions, colored by cluster labels.
#     """
#     labels = df['label'].values

#     # Dimensionality Reduction with UMAP
#     reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
#     umap_embeddings = reducer.fit_transform(embeddings)

#     # Clustering with KMeans
#     kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
#     kmeans.fit(embeddings)
#     cluster_labels = kmeans.labels_

#     # Create a color array with bright green for matches
#     colors = []
#     for i in range(len(cluster_labels)):
#         if cluster_labels[i] == labels[i]:
#             colors.append("lime")  # Bright green for matches
#         else:
#             colors.append(cluster_labels[i])  # Use the cluster label for other points

#     # Visualization with Plotly
#     fig = px.scatter_3d(
#         x=umap_embeddings[:, 0],
#         y=umap_embeddings[:, 1],
#         z=umap_embeddings[:, 2],
#         color=colors,
#         title="3D UMAP Projection of Semantic Embeddings",
#         labels={"color": "Cluster Label"}
#     )
#     fig.update_layout(
#         scene=dict(
#             xaxis_title="UMAP Dimension 1",
#             yaxis_title="UMAP Dimension 2",
#             zaxis_title="UMAP Dimension 3"
#         ),
#         coloraxis_colorbar=dict(title='Cluster Label'),
#         template='plotly_dark',
#         height=800,
#         width=1000
#     )

#     return fig

def analyze_embeddings_mark(df, embeddings, n_neighbors=25, min_dist=0.01, n_components=3, random_state=42):
    """
    Analyzes and visualizes semantic embeddings using UMAP for dimensionality reduction.

    Parameters:
    - df: DataFrame containing 'label' column.
    - embeddings: The high-dimensional embeddings array.
    - n_neighbors, min_dist, n_components: UMAP parameters for dimensionality reduction.
    - random_state: Seed for reproducibility.

    Returns:
    - A Plotly figure object visualizing the embeddings in reduced dimensions, colored by labels.
    """
    labels = df['label'].values

    # Dimensionality Reduction with UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    umap_embeddings = reducer.fit_transform(embeddings)

    # Visualization with Plotly
    fig = px.scatter_3d(
        x=umap_embeddings[:, 0],
        y=umap_embeddings[:, 1],
        z=umap_embeddings[:, 2],
        color=labels,
        title="3D UMAP Projection of Semantic Embeddings",
        labels={"color": "Label"}
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3"
        ),
        coloraxis_colorbar=dict(title='Label'),
        template='plotly_dark',
        height=800,
        width=1000
    )

    return fig