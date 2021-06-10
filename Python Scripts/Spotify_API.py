import spotipy
import json
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go

def print_class(instance):
    keys = []
    class_var = instance.__dict__
    for k in class_var:
        keys.append(k)
    for j in keys:
        if type(class_var) is not str: 
            print(j + ' :    ' + str(class_var[j]))
        else:
            print(j + ' :    ' + class_var[j])
    return None

def get_str_class(instance):
    keys, r = [], ''
    class_var = instance.__dict__
    for k in class_var:
        keys.append(k)
    for j in keys:
        if type(class_var) is not str: 
            r += j + ' :    ' + str(class_var[j]) + "\n"
        else:
            r += j + ' :    ' + str(class_var[j]) + "\n"
    return r

def print_json(object):
    print(json.dumps(object, sort_keys = True, indent = 4))
    return None

def View_All(class_list):
    for i in range(0, len(class_list)):
        print('INDEX: ' + str(i) + '\n')
        print_class(class_list[i])
        print('\n')

class album:
    def __init__(self):
        self.href = ''
        self.uri = ''
        self.id = ''
        self.album_name = ''
        self.artist_name = ''
        self.release_date = ''
        self.tracks = ''
        self.images = ''
        self.url = ''

class artist:
    def __init__(self, name):
        self.external_urls = ''
        self.uri = ''
        self.id = ''
        self.input_name = name
        self.spotify_name = ''
        self.followers = 0
        self.genres = ''
        self.popularity = 0
        self.href = ''
        self.images = ''

    def Artist_Profile(self, tp_art):
        self.external_urls = tp_art['external_urls']
        self.uri = tp_art['uri']
        self.id = tp_art['id']
        self.spotify_name = tp_art['name']
        self.followers = tp_art['followers']['total']
        self.genres = tp_art['genres']
        self.popularity = tp_art['popularity']
        self.href = tp_art['href']
        self.images = tp_art['images']

    def Get_Artist(self):
        global client
        global sp 
        if client == None and sp == None:
            client = SpotifyClientCredentials(client_id = cid, client_secret = secret)
            sp = spotipy.Spotify(client_credentials_manager = client)
        results = sp.search(q='artist:' + self.input_name, type='artist')
        List_of_Artists = results['artists']
        Attributes_Artists = List_of_Artists['items']
        Top_Artist = Attributes_Artists[0]
        self.Artist_Profile(Top_Artist)

class song:
    def __init__(self):
        self.album_title = ''
        self.release_date = ''
        self.track_total = 0
        self.artist_name = ''
        self.disc_number = 0
        self.duration_ms = 0
        self.explicit = ''
        self.track_url = ''
        self.href = ''
        self.track_id = ''
        self.is_local = ''
        self.track_popularity = 0
        self.track_number = 0
        self.uri = ''

class track:
    def __init__(self):
        self.disc_number = 0
        self.duration_ms = 0
        self.explicit = ''
        self.track_url = ''
        self.href = ''
        self.track_id = ''
        self.is_local = ''
        self.track_name = ''
        self.track_number = 0
        self.uri = ''

class track_features:
    def __init__(self):
        self.acousticness = ''
        self.analysis_url = ''
        self.danceability = ''
        self.energy = ''
        self.id = ''
        self.instrumentalness = ''
        self.key = ''
        self.liveness = ''
        self.loudness = ''
        self.mode = ''
        self.speechiness = ''
        self.tempo = ''
        self.time_signature = ''
        self.track_href = ''
        self.uri = ''
        self.valence = ''

class Rec_Album:
    def __init__(self):
        self.artist_external_url = ''
        self.artist_href = ''
        self.artist_id = ''
        self.artist_name = ''
        self.artist_uri = ''
        self.album_external_url = ''
        self.album_href = ''
        self.album_id = ''
        self.album_images = ''
        self.album_name = ''
        self.album_release_date = ''
        self.album_total_tracks = ''
        self.album_uri = ''

class Rec_Song:
    def __init__(self):
        self.artist_name = ''
        self.album_url = ''
        self.album_id = ''
        self.album_title = ''
        self.album_uri = ''
        self.song_url = ''
        self.song_id = ''
        self.song_name = ''
        self.song_popularity = 0
        self.track_number = 0
        self.song_uri = ''

def Get_Albums(uri, type):
    album_ClassList = []
    artist_albums = sp.artist_albums(uri, album_type = type)
    available_albums = artist_albums['items']
    k = 0
    while k < len(available_albums):
        album_stats = album()
        album_stats.href = available_albums[k]['href']
        album_stats.uri = available_albums[k]['uri']
        album_stats.id = available_albums[k]['id']
        album_stats.album_name = available_albums[k]['name']
        album_stats.artist_name = available_albums[k]['artists'][0]['name']
        album_stats.release_date = available_albums[k]['release_date']
        album_stats.tracks = available_albums[k]['total_tracks']
        album_stats.images = available_albums[k]['images']
        album_stats.url = available_albums[k]['external_urls']['spotify']
        album_ClassList.append(album_stats)
        del album_stats
        k += 1
    return album_ClassList

def Get_Song(track):
    global client
    global sp 
    if client == None and sp == None:
        client = SpotifyClientCredentials(client_id = cid, client_secret = secret)
        sp = spotipy.Spotify(client_credentials_manager = client)
    song_stats = song()
    songs = sp.search(track, limit=3)
    single = songs['tracks']['items'][0]
    song_stats.album_title = single['album']['name']
    song_stats.release_date = single['album']['release_date']
    song_stats.track_total = single['album']['total_tracks']
    song_stats.artist_name = single['artists'][0]['name']
    song_stats.disc_number = single['disc_number']
    song_stats.duration_ms = single['duration_ms']
    song_stats.explicit = single['explicit']
    song_stats.track_url = single['external_urls']['spotify']
    song_stats.href = single['href']
    song_stats.track_id = single['id']
    song_stats.is_local = single['is_local']
    song_stats.track_popularity = single['popularity']
    song_stats.track_number = single['track_number']
    song_stats.uri = single['uri']
    return song_stats

def Get_Album_Tracks(album_uri):
    album_TrackList = []
    album_tracks_complete = sp.album_tracks(album_uri)
    all_tracks = album_tracks_complete['items']
    i = 0
    while i < len(all_tracks):
        track_stats = track()
        track_stats.disc_number = all_tracks[i]['disc_number']
        track_stats.duration_ms = all_tracks[i]['duration_ms']
        track_stats.explicit = all_tracks[i]['explicit']
        track_stats.track_url = all_tracks[i]['external_urls']['spotify']
        track_stats.href = all_tracks[i]['href']
        track_stats.track_id = all_tracks[i]['id']
        track_stats.is_local = all_tracks[i]['is_local']
        track_stats.track_name = all_tracks[i]['name']
        track_stats.track_number = all_tracks[i]['track_number']
        track_stats.uri = all_tracks[i]['uri']
        album_TrackList.append(track_stats)
        del track_stats
        i += 1
    return album_TrackList

def Album_Generator(artist_ID):
    recommended_albums_list = []
    recommended_albums = sp.recommendations(seed_artists = [artist_ID])
    all_recommended_albums = recommended_albums['tracks']
    for i in range(0, len(all_recommended_albums)):
        rec_alb = Rec_Album()
        recommended_album = recommended_albums['tracks'][i]['album']
        rec_alb.artist_external_url = recommended_album['artists'][0]['external_urls']['spotify']
        rec_alb.artist_href = recommended_album['artists'][0]['href']
        rec_alb.artist_id = recommended_album['artists'][0]['id']
        rec_alb.artist_name = recommended_album['artists'][0]['name']
        rec_alb.artist_uri = recommended_album['artists'][0]['uri']
        rec_alb.album_external_url = recommended_album['external_urls']
        rec_alb.album_href = recommended_album['href']
        rec_alb.album_id = recommended_album['id']
        rec_alb.album_images = recommended_album['images']
        rec_alb.album_name = recommended_album['name']
        rec_alb.album_release_date = recommended_album['release_date']
        rec_alb.album_total_tracks = recommended_album['total_tracks']
        rec_alb.album_uri = recommended_album['uri']
        recommended_albums_list.append(rec_alb)
    return recommended_albums_list

def Song_Features(single):
    features = sp.audio_features(single)
    track_stats = track_features()
    track_stats.acousticness = features[0]['acousticness']
    track_stats.danceability = features[0]['danceability']
    track_stats.duration_ms = features[0]['duration_ms']
    track_stats.energy = features[0]['energy']
    track_stats.id = features[0]['id']
    track_stats.instrumentalness = features[0]['instrumentalness']
    track_stats.key = features[0]['key']
    track_stats.liveness = features[0]['liveness']
    track_stats.loudness = features[0]['loudness']
    track_stats.mode = features[0]['mode']
    track_stats.speechiness = features[0]['speechiness']
    track_stats.tempo = features[0]['tempo']
    track_stats.time_signature = features[0]['time_signature']
    track_stats.track_href = features[0]['track_href']
    track_stats.type = features[0]['type']
    track_stats.uri = features[0]['uri']
    track_stats.valence = features[0]['valence']
    return track_stats
   
def Song_Generator(artist_ID):
    recommended_songs_list = []
    recommended_songs = sp.recommendations(seed_artists = [artist_ID])
    all_recommended_songs = recommended_songs['tracks']
    for k in range(0, len(all_recommended_songs)):
        rec_song = Rec_Song()
        rec_song.artist_name = recommended_songs['tracks'][k]['album']['artists'][0]['name']
        rec_song.album_url = recommended_songs['tracks'][k]['album']['external_urls']['spotify']
        rec_song.album_id = recommended_songs['tracks'][k]['album']['id']
        rec_song.album_title = recommended_songs['tracks'][k]['album']['name']
        rec_song.album_uri = recommended_songs['tracks'][k]['album']['uri']
        rec_song.song_url = recommended_songs['tracks'][k]['external_urls']['spotify']
        rec_song.song_id = recommended_songs['tracks'][k]['id']
        rec_song.song_name = recommended_songs['tracks'][k]['name']
        rec_song.song_popularity = recommended_songs['tracks'][k]['popularity']
        rec_song.track_number = recommended_songs['tracks'][k]['track_number']
        rec_song.song_uri = recommended_songs['tracks'][k]['uri']
        recommended_songs_list.append(rec_song)
    return recommended_songs_list

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def plotter():
    rec_song_uris = []
    rec_song_popularity = []
    recommended_songs = Song_Generator(artist_stats.id)
    for k in range(0, len(recommended_songs)):
        rec_song_uris.append(recommended_songs[k].song_uri)
        rec_song_popularity.append(recommended_songs[k].song_popularity)
    all_song_analysis = []
    danceability, energy, instrumentalness, liveness, loudness, tempo,  = [], [], [], [], [], []
    for j in range(0, len(rec_song_uris)):
        song_analysis = Song_Features(rec_song_uris[j])
        all_song_analysis.append(song_analysis)
        danceability.append(song_analysis.danceability)
        energy.append(song_analysis.energy)
        instrumentalness.append(song_analysis.instrumentalness)
        liveness.append(song_analysis.liveness)
        loudness.append(song_analysis.loudness)
        tempo.append(song_analysis.tempo)
    df = pd.DataFrame({'Danceability':danceability, 'Energy':energy, 'Instrumentalness':instrumentalness, 'Liveness':liveness, 'Loudness':loudness, 'Tempo':tempo})
    print(df)
    df = normalize(df)
    categories = ['Dance', 'Energy', 'Instrumental', 'Liveliness', 'Loudness', 'Tempo']
    categories += categories[:1]
    trends = [df['Danceability'].mean(), df['Energy'].mean(), df['Instrumentalness'].mean(), df['Liveness'].mean(), df['Loudness'].mean(), df['Tempo'].mean()]
    trends += trends[:1]
    fig = go.Figure(
        data=[
            go.Scatterpolar(r = trends, theta=categories, fill = 'toself', name = 'Music Trends')
        ],
        layout = go.Layout(
            title = go.layout.Title(text = ''),
            polar = dict(
                radialaxis = dict(
                    visible = True,
                    range = [0, 1]
                )
            ),
            showlegend = False
        )
    )
    plt_div = plot(fig, output_type='div')
    return plt_div

# 1 CHOOSE ARTIST AND PRINT CLASS
# ````````````````````````````````

client, sp = None, None
cid = ''        # Insert cid here
secret = ''     # Insert secret here

artist_name = 'Journey'
artist_stats = artist(artist_name)
artist_stats.Get_Artist()
# print_class(artist_stats)

# 2 GET ALL ALBUMS
# ````````````````````````````````

# Albums = Get_Albums(artist_stats.uri, 'album')
# View_All(Albums)   # Remove comment to print all albums.  Choose index for album of choice. 

# 3 GET SONG (all of step 4 must be commented to run step 3...for now)
# ````````````````````````````````

# track = Get_Song('Faithfully')
# print_class(track)   # Remove comment to print song profile

# 4 GET ALL SONGS FROM AN ALBUM (all of step 3 must be commented to run step 4...for now)
# ````````````````````````````````

# Tracks = Get_Album_Tracks(Albums[0].uri)   # insert album index from step 2 here, step 2 code must not be commented for step 4 to work.
# View_All(Tracks)    # Remove comment to print all tracks

# 5 GET SONGS AUDIO FEATURES
# ````````````````````````````````

# song_analysis = Song_Features(track.uri)    # insert track index from step 3 here, step 3 code must not be commented for step 5 to work.
# print_class(song_analysis)

# 6 GENERATE RECOMMENDED ALBUM LIST
# ````````````````````````````````

# recommended_albums = Album_Generator(artist_stats.id)
# for k in range(0, len(recommended_albums)):
#     print_class(recommended_albums[k])
#     print('\n')

# 7 GENERATE RECOMMENDED ALBUM LIST
# ````````````````````````````````

# recommended_songs = Song_Generator(artist_stats.id)
# for k in range(0, len(recommended_songs)):
#     print_class(recommended_songs[k])
#     print('\n')





# INSTRUCTIONS FOR GETTING STARTED

# 1. Visit https://developer.spotify.com/dashboard/
# 2. Create an account, login, and obtain your cid and secret by creating a project in your profile.
#         https://cran.r-project.org/web/packages/spotidy/vignettes/Connecting-with-the-Spotify-API.html
# 3. Insert you cid and secret into the variables on lines 358 and 359.
# 4. Import the necessary modules located at the top of the script.
# 5. Choose an artist on line 361.
# 6. Uncomment line 364 and execute the program.
