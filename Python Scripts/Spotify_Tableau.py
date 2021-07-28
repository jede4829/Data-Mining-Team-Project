# ----------------------------------------------------------------#
# Import Modules and Client
# ----------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
import Spotify_Def

# ----------------------------------------------------------------#
# Import Data File (Step 1: Choose Data File)
# ----------------------------------------------------------------#

data_file = 'countryartists.csv'
dataset = Spotify_Def.read_file(data_file)
acoustics = dataset.iloc[:, 6:]

# ----------------------------------------------------------------#
# Normalize features that have different scales
# ----------------------------------------------------------------#

norm = MinMaxScaler()
normalized_dataset = norm.fit_transform(dataset[['Tempo', 'Key', 'Duration','Loudness', 'Popularity']]) 
dataset[['Tempo', 'Key', 'Duration','Loudness', 'Popularity']] = normalized_dataset
df_acoustics = dataset[['Duration','Explicit','Acousticness','Danceability','Energy','Instrumentalness','Key','Liveness','Loudness','Mode','Speechiness','Tempo','Time_Signature','Valence']]

# ----------------------------------------------------------------#
# SNS Correlation Plot of Acoustic Pairs (Step 2: Generate Plot)
# ----------------------------------------------------------------#

def linear_reg_plot(xaxis, yaxis, scatter_color, line_color, scatter_weight):
    plt.figure(figsize = (10, 8))
    sns.set(style = "whitegrid")
    sns.regplot(x = df_acoustics[xaxis], y = df_acoustics[yaxis], data = df_acoustics, scatter_kws = {'color': scatter_color, 'alpha': scatter_weight}, line_kws = {'color':line_color})
    plt.title(xaxis + ' x ' + yaxis)
    plt.show()

linear_reg_plot('Energy', 'Loudness', 'blue', 'red', 0.4)  # Choose two acoustic Characteristics (for this review the csv file in step 1 and select two different columns), colors, and scatter plot weight.