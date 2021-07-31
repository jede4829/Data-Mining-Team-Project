from tracks_plots import *
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def rm_strcol(df):
    df = df[[col for col in df if col not in ['id', 'name', 'artists', 'id_artists']]]
    return df

def kmeans(df, attr):
    mat = df.values
    km = sklearn.cluster.KMeans(n_clusters=9)
    km.fit(mat)
    centroids = km.cluster_centers_
    #plt.scatter(df[attr], df['popularity'], c= km.labels_.astype(float), s=50, alpha=0.5)
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='white', s=50)
    #plt.show()
    labels = km.labels_
    df['cluster']=labels
    return df

def bar_plots(df, attr):
    x, y = 0, 0
    fig, axes = plt.subplots(4, 4, figsize=(10, 8), sharey=True)
    plt.tight_layout(pad=4.0)
    sns.set_theme(color_codes=True)
    for i in range(len(attr)):
        sns.barplot(x=df['cluster'], y=df[attr[i]], data=df,ax=axes[x,y]);
        axes[x,y].set_title(str(attr[i]))
        if y == 3:
            y = 0
            x += 1
        else:
            y += 1
        
    fig.suptitle('Cluster averages')
    plt.show()

def normalize(df, attrs):
    norm = MinMaxScaler()
    dataset = norm.fit_transform(df[attrs]) 
    df[attrs] = dataset
    return df

def main():
    to_normalize = ['release_date','time_signature','tempo', 'key', 'duration_ms',
                    'loudness', 'popularity']
    attrs = ['popularity', 'release_date', 'danceability', 'energy', 'key',
             'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness',
             'liveness','valence', 'tempo', 'time_signature','duration_ms', 'explicit']
    df = read_file("tracks.csv")
    df = clean_dates(df)
    df = clean_data(df)
    df = sample_size(df, 5000)
    df = rm_strcol(df)
    df = normalize(df, to_normalize)
    df = kmeans(df, 'release_date')
    df_cluster = pd.DataFrame(np.array([i for i in range(9)]), columns=['cluster'])
    df_list = [[] for i in range(9)]
    for x in range(9):
        dfc = df.loc[df['cluster'] == x]
        for i in range(len(attrs)):
            df_list[x].append(dfc[attrs[i]].mean())
    df_avg = pd.DataFrame(np.array(df_list), columns=attrs)
    df_avg['cluster'] = df_cluster['cluster']
    print(df_avg)
    bar_plots(df_avg, attrs)
    
    
main()
