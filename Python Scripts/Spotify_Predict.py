
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
from sklearn.neighbors import KNeighborsClassifier
norm = MinMaxScaler()

# ----------------------------------------------------------------#
# Print New Line, Read CSV File into DataFrame
# ----------------------------------------------------------------#

def new_line(): print('\n')
def read_file(csv_file): return pd.read_csv(csv_file)

# ----------------------------------------------------------------#
# Preview and Summarize Data Set
# ----------------------------------------------------------------#

def describe_dataset(set):
    new_line()
    print(set.head())
    new_line()
    print(set.describe())
    new_line()

# ----------------------------------------------------------------#
# Import Data File to Train Prediction Algorithm
# ----------------------------------------------------------------#

def Read_Training_Data(mode):
    if mode == 1: set = read_file('country.csv')
    elif mode == 2:
        set = read_file('tracks.csv')
        hit, popularity = [], set['popularity']
        for i in range(0, len(popularity)):
            if popularity[i] >= 50: hit.append(1)
            else: hit.append(0)
        set['Hit?'] = hit
        return set
    elif mode == 3:
        set = pd.read_csv('spotifyAnalysis-08022020.csv')
        return set

# ----------------------------------------------------------------#
# Normalize features that have different scales
# ----------------------------------------------------------------#

def Normalizer(set, mode):
    if mode == 1:
        normalized_dataset = norm.fit_transform(set[['Tempo', 'Key', 'Duration','Loudness', 'Popularity']]) 
        set[['Tempo', 'Key', 'Duration','Loudness', 'Popularity']] = normalized_dataset
    elif mode == 2:
        normalized_dataset = norm.fit_transform(set[['popularity','duration_ms','key','loudness','tempo']]) 
        set[['popularity','duration_ms','key','loudness','tempo']] = normalized_dataset
    elif mode == 3:
        scaled_values = norm.fit_transform(set[['tempo', 'key', 'duration_ms','loudness', 'popularity']]) 
        set[['tempo', 'key', 'duration_ms','loudness', 'popularity']] = scaled_values
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
# Analyze Hit Songs in Test Set
# ----------------------------------------------------------------#

def num_of_hits(set):
    df_hits = set['Hit?'].value_counts().reset_index()
    df_hits.columns = ['Hit?', 'count']
    print('\n')
    print('Number of hits in testset:\t\t', df_hits.loc[0,'count'])
    print('Number of non_hits in testset:\t\t', df_hits.loc[1,'count'])
    print('\n')
    return int(df_hits.loc[0,'count']), int(df_hits.loc[1,'count'])

# ----------------------------------------------------------------#
# Analyze Artists with hit Songs in Test Set
# ----------------------------------------------------------------#

def num_hits_by_artist(set):
    top_artists = set['artist'].value_counts()  #[:30], append and change number to truncate results
    plt.figure(1 , figsize = (15, 12))
    ax = sns.barplot(x = list(top_artists.index), y = list(top_artists.values), palette = "Purples_d")
    ax.set_xticklabels(list(top_artists.index), rotation = 90)
    ax.set_ylabel('Number of Hits')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------#
# Summarize Acoustics of Songs in Solution Set
# ----------------------------------------------------------------#

def acoustics_analysis(set):
    if mode == 1: acoustic_params = ['Acousticness', 'Danceability', 'Energy', 'Key', 'Liveness', 'Loudness', 'Tempo', 'Valence']
    elif mode == 2: acoustic_params = ['acousticness', 'danceability', 'energy', 'key', 'liveness', 'loudness', 'tempo', 'valence']
    for i in range(0, len(acoustic_params)):
        temp, above, below = list(set[acoustic_params[i]]), 0, 0
        for j in range(0, len(temp)):
            if temp[j] >= 0.5: above += 1
            elif temp[j] < 0.5: below += 1
        print('Songs with high ' + acoustic_params[i] + ' = ' + str(above))
        print('Songs with low ' + acoustic_params[i] + ' = ' + str(below))

# ----------------------------------------------------------------#
# Set Mode, Read CSV File, Show Selected Results
# ----------------------------------------------------------------#

mode = 2    # 1 for custom, 2 for tracks.csv
heatmap = False
linear = False
describe_train_data = False
describe_test_data = False
nhits = False
nhitsartist = False
acoustics = False
hits_num, nhits_num = 0, 0
predict = True

dataset = Read_Training_Data(mode)
if mode == 2:
    testset = read_file('test.csv')
if mode == 3:
    testset = read_file('predictSpotifyAnalysis-08022020.csv')
if describe_train_data:
    describe_dataset(dataset)
if describe_test_data:
    describe_dataset(testset)
dataset = Normalizer(dataset, mode)
# dataset = dataset.drop(['explicit'], axis = 1)
testset = Normalizer(testset, mode)

if mode == 1 and heatmap:
    heatmap_plot(dataset.drop(['Hit?', 'Artist', 'Song', 'Artist Genre', 'Popularity'], axis = 1), 'Greens')
if mode == 2 and heatmap:
    heatmap_plot(dataset.drop(['id', 'name', 'Hit?', 'popularity', 'artists', 'id_artists', 'release_date'], axis = 1), 'Greens')
if mode == 1 and linear:
    linear_reg_plot(dataset, 'Energy', 'Loudness', 'blue', 'red', 0.1)
if mode == 2 and linear:
    linear_reg_plot(dataset, 'energy', 'loudness', 'blue', 'red', 0.1)
if nhits:
    hits_num, nhits_num = num_of_hits(testset)
if nhitsartist:
    num_hits_by_artist(testset)
if acoustics:
    acoustics_analysis(testset)

# ----------------------------------------------------------------#
# Create Machine Learning Model
# ----------------------------------------------------------------#

if predict:
    if mode == 2:
        # print(dataset.drop(['id','name','Hit?','artists','id_artists','release_date','popularity'], axis = 1))
        X = dataset.drop(['id','name','Hit?','artists','id_artists','release_date','popularity'], axis = 1).values
        y = dataset[['Hit?']].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    elif mode == 3:
        X = dataset.drop(['success', 'artist', 'track_name'], axis=1).values
        Y = dataset[['success']].values
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 0)

    predict_mode = 2    # 1 for logistic regression, 2 for random forest, 3 for SVM, 4 for KNN

    if predict_mode == 1:
        lg_model = LogisticRegression(solver = 'sag', max_iter=200, class_weight='balanced')
        lg_model.fit(x_train, y_train.ravel())
        y_pred = lg_model.predict(x_test)
    elif predict_mode == 2:
        rf_model = RandomForestClassifier(n_estimators = 20, max_depth = 6, class_weight = 'balanced_subsample')
        rf_model.fit(x_train, y_train.ravel())
        y_pred = rf_model.predict(x_test)
    elif predict_mode == 3:
        svm_model = svm.SVC(kernel='linear')
        svm_model.fit(x_train, y_train.ravel())
        y_pred = svm_model.predict(x_test)
    elif predict_mode == 4:
        knn = KNeighborsClassifier(n_neighbors=3,p=2,metric='euclidean')
        knn.fit(x_train, y_train.ravel())
        y_pred = knn.predict(x_test)

    new_line()
    print('Accuracy: ', accuracy_score(y_test, y_pred))

# ----------------------------------------------------------------#
# Predict which songs will be hits
# ----------------------------------------------------------------#

df_test = testset.drop(['artist','artist Genre','song','Hit?','popularity'], axis=1).values
if predict_mode == 1: test_predict = lg_model.predict(df_test)
if predict_mode == 2: test_predict = rf_model.predict(df_test)
if predict_mode == 3: test_predict = svm_model.predict(df_test)
if predict_mode == 4: test_predict = knn.predict(df_test)

hits_predict = (test_predict == 1).sum()
print(hits_predict, "out of", len(test_predict), "was predicted as HIT")
new_line

df = pd.DataFrame({'Song': testset['song'], 'Artist': testset['artist'], 'Predict': test_predict})
df.sort_values(by=['Predict'], inplace=True, ascending=False)
print(df)
