import seaborn as sns
import pandas as pd

points = pd.read_csv("assignments_2.txt", header=None)
centroids = pd.read_csv("centroids_2.txt", header=None)

points['is_centroid'] = 1
centroids['is_centroid'] = 0

df = pd.concat([points, centroids])
assignments_col = len(df.columns) - 2

sns.scatterplot(x=df[0], y=df[1], hue=df[assignments_col], palette="tab10", size=df['is_centroid'])