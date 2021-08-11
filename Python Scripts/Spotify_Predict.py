
# ----------------------------------------------------------------#
# Import Modules and Client
# ----------------------------------------------------------------#

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()

# ----------------------------------------------------------------#
# Print New Line, Read CSV File into DataFrame
# ----------------------------------------------------------------#

def new_line(): print('\n')
def read_file(csv_file): return pd.read_csv(csv_file)

# ----------------------------------------------------------------#
# Import Data File to Train Prediction Algorithm
# ----------------------------------------------------------------#

def Read_Training_Data(mode):
    if mode == 1: 
        set = read_file('top100.csv')
    elif mode == 2:
        set = read_file('tracks.csv')
        hit, popularity = [], set['popularity']
        for i in range(0, len(popularity)):
            if popularity[i] >= 50: hit.append(1)
            else: hit.append(0)
        set['hit'] = hit
    return set

# ----------------------------------------------------------------#
# Normalize features that have different scales
# ----------------------------------------------------------------#

def Normalizer(set, mode):
    if mode == 1:
        normalized_dataset = norm.fit_transform(set[['Danceability','Energy','Key','Loudness','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Time_signature']]) 
        set[['Danceability','Energy','Key','Loudness','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Time_signature']] = normalized_dataset
    elif mode == 2:
        normalized_dataset = norm.fit_transform(set[['danceability','energy','key','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature']]) 
        set[['danceability','energy','key','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature']] = normalized_dataset
        return set
    return set

# ----------------------------------------------------------------#
# SNS Heat Map
# ----------------------------------------------------------------#

def heatmap_plot(df, cmap_color):
    plt.figure(figsize = (14,12))
    mask = np.zeros_like(df.corr(), dtype = bool)
    mask[np.triu_indices_from(mask, 1)] = True
    sns.heatmap(df.corr(), mask = mask, annot = True, cmap = cmap_color)
    plt.show()

# ----------------------------------------------------------------#
# SNS Correlation Plot of Acoustic Pairs
# ----------------------------------------------------------------#

def linear_reg_plot(set, xaxis, yaxis, scatter_color, line_color, scatter_weight):
    plt.figure(figsize = (10, 8))
    sns.set(style = "whitegrid")
    sns.regplot(x = set[xaxis], y = set[yaxis], data = set, scatter_kws = {'color': scatter_color, 'alpha': scatter_weight}, line_kws = {'color':line_color})
    plt.title(xaxis + ' x ' + yaxis)
    plt.show()

# ----------------------------------------------------------------#
# Summarize Acoustics of Songs in Solution Set
# ----------------------------------------------------------------#

def genres(set):
    genres = []
    cid = '9cab75c094f941978bff0389a9d5dfa4'
    secret = '20b6077267ae4241af1756d4544b5069'
    client = SpotifyClientCredentials(client_id = cid, client_secret = secret)
    sp = spotipy.Spotify(client_credentials_manager = client)
    temp = list(set.loc[:,'name'])
    for k in range(0, len(temp)):
        result = sp.search(temp[k])
        track = result['tracks']['items'][0]
        artist = sp.artist(track["artists"][0]["external_urls"]["spotify"])
        genres.append(artist["genres"])
    return genres

# ----------------------------------------------------------------#
# Set Mode, Read CSV File, Show Selected Results
# ----------------------------------------------------------------#

mode = 2    # 1 for custom, 2 for tracks.csv
heatmap = False
linear = False
get_genres = False
predict = True

dataset = Read_Training_Data(mode)
if mode == 2:
    dataset = Normalizer(dataset, mode)
    dataset = dataset[['name','popularity','duration_ms','release_date','explicit','danceability','energy','key','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature','hit']]
    dataset['popularity'] = dataset['popularity'].astype(int)
    dataset = dataset.loc[dataset.loc[:,'popularity'] >= 20]
    dataset['duration_ms'] = dataset['duration_ms'].astype(int)
    dataset = dataset.loc[(dataset.loc[:,'duration_ms'] >= 120000)&(dataset.loc[:,'duration_ms'] <= 300000)]
    dataset['release_date'] = dataset['release_date'].str.split('-').str[0].astype(int)
    dataset = dataset.loc[(dataset.loc[:,'release_date'] > 2018)&(dataset.loc[:,'release_date'] < 2022)]
    if get_genres:
        genre_column = genres(dataset)
        dataset['genres'] = genre_column
        print(dataset.head())

new_line()
print('Number of Songs: ' + str(len(dataset)))

if mode == 1 and heatmap: heatmap_plot(dataset.drop(['Hit?', 'Artist', 'Song', 'Artist Genre', 'Popularity'], axis = 1), 'Oranges')
if mode == 2 and heatmap: heatmap_plot(dataset.drop(['name', 'hit', 'popularity', 'release_date'], axis = 1), 'Oranges')
if mode == 1 and linear: linear_reg_plot(dataset, 'Energy', 'Loudness', 'blue', 'red', 0.1)
if mode == 2 and linear: linear_reg_plot(dataset, 'explicit', 'speechiness', 'brown', 'yellow', 0.1)

# ----------------------------------------------------------------#
# Create Machine Learning Model
# ----------------------------------------------------------------#

if predict:
    if mode == 2:
        X, y = dataset[['explicit','danceability','energy','key','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']].values, dataset[['hit']].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
    predict_mode = 1    # 1 for logistic regression, 2 for random forest, 3 for KNN
    if predict_mode == 1:
        lg_model = LogisticRegression()
        lg_model.fit(x_train, y_train.ravel())
        y_pred = lg_model.predict(x_test)
    elif predict_mode == 2:
        rf_model = RandomForestClassifier(n_estimators = 5, max_depth = 3)
        rf_model.fit(x_train, y_train.ravel())
        y_pred = rf_model.predict(x_test)
    elif predict_mode == 3:
        knn = KNeighborsClassifier(n_neighbors = 17, p = 2, algorithm = 'auto', metric = 'minkowski', weights = 'distance')
        knn.fit(x_train, y_train.ravel())
        y_pred = knn.predict(x_test)

    new_line()
    print(classification_report(y_test, y_pred))
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))
    new_line()
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(cm)
    new_line()
