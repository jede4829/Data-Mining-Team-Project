import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import random
from random import randint
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

to_normalize = ['release_date','time_signature','tempo', 'key', 'duration_ms',
                    'loudness', 'popularity']
attrs = ['popularity', 'release_date', 'danceability', 'energy', 'key',
             'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness',
             'liveness','valence', 'tempo', 'time_signature','duration_ms', 'explicit']
    
def normalize(df, attrs):
    norm = MinMaxScaler()
    dataset = norm.fit_transform(df[attrs]) 
    df[attrs] = dataset
    return df

def correlation(attribute1, attribute2, file):
    #calculates correlation coefficient
    attr1 = file[attribute1]
    attr2 = file[attribute2]
    try:
        correlation = attr1.corr(attr2)
        return correlation
    except:
        return

def sample_size(df, size):
    return df.sample(n=size)

def clean_data(df):
    #eliminating 0 popularity and erroneous data
    df = df.loc[df['release_date']!=1900]
    df = df.loc[df['popularity']>0]
    return df

def fix_order(df):
    #moving characteristics to end
    df = df[[col for col in df if col not in ['popularity', 'duration_ms', 'explicit']] 
       + ['popularity', 'duration_ms', 'explicit']]
    return df
    
def clean_dates(df):
    #converting date to year only
    df['release_date'] = df['release_date'].str.replace('-','').str[0:4]
    df['release_date'] = pd.to_numeric(df['release_date'])
    return df

def clean_genres(df):
    #tracks csv does not contain genres, only artists csv
    #to access a single row's list of genres:
    #lst = df['genres'].loc[28676:28676][28676]
    df['genres'] = df['genres'].str.replace("'","").str.replace(', ',',')
    df['genres'] = df['genres'].str.strip('[]').str.split(",")
    return df
    
def read_file(fileName):
    file = pd.read_csv(fileName)
    return file

def get_headings(file):
    headings = list(file.columns)
    return headings
