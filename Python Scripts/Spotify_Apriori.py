
# ----------------------------------------------------------------#
# Import Modules and Client
# ----------------------------------------------------------------#

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import spotipy
import Spotify_Class
import Spotify_Def
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------------------------------------------------------#
# Retrieve recommended songs based off artist
# ----------------------------------------------------------------#

def Song_Generator(artist_ID):
    song_list = []
    artist_name, song_name, popularity, track_number, uri = [], [], [], [], []
    songs = sp.recommendations(seed_artists = [artist_ID], limit = 5)
    recommended_songs = songs['tracks']
    for k in range(0, len(recommended_songs)):
        rec_song = Spotify_Class.song()
        rec_song.artist_name = songs['tracks'][k]['album']['artists'][0]['name']
        artist_name.append(songs['tracks'][k]['album']['artists'][0]['name'])
        rec_song.song_url = songs['tracks'][k]['external_urls']['spotify']
        rec_song.song_id = songs['tracks'][k]['id']
        rec_song.song_name = songs['tracks'][k]['name']
        song_name.append(songs['tracks'][k]['name'])
        rec_song.song_popularity = songs['tracks'][k]['popularity']
        popularity.append(songs['tracks'][k]['popularity'])
        rec_song.track_number = songs['tracks'][k]['track_number']
        track_number.append(songs['tracks'][k]['track_number'])
        rec_song.song_uri = songs['tracks'][k]['uri']
        uri.append(songs['tracks'][k]['uri'])
        song_list.append(rec_song)
    return song_list, artist_name, song_name, popularity, track_number, uri

# ----------------------------------------------------------------#
# Spotify API Collect List of Song Recommendations
# ----------------------------------------------------------------#

def song_recommendations(profile, loop):
    rec_compilation, df_rec_compilation = [], []
    artist_data, song_name_data, popularity_data, track_numbers_data, uris_data = [], [], [], [], []
    for k in range(0, loop):
        song_list, artist_names, song_names, popularity, track_numbers, uris = Song_Generator(profile.id)
        rec_compilation.append(song_list)
        artist_data.append(artist_names)
        song_name_data.append(song_names)
        popularity_data.append(popularity)
        track_numbers_data.append(track_numbers)
        uris_data.append(uris)
        df_recommendations = pd.DataFrame({'Artist':artist_names, 'Song':song_names, 'Popularity':popularity, 'Track Number':track_numbers, 'URI':uris})
        df_rec_compilation.append(df_recommendations)
    return rec_compilation, df_rec_compilation, artist_data, song_name_data, popularity_data, track_numbers_data, uris_data

# ----------------------------------------------------------------#
# Generate Frequent Item Sets Using Apriori Algorithm
# ----------------------------------------------------------------#

def Apriori(data_set):
    te = TransactionEncoder()
    te_ary = te.fit(data_set).transform(data_set)
    df = pd.DataFrame(te_ary, columns = te.columns_)
    item_sets = apriori(df, min_support = 0.2, use_colnames = True)
    print(item_sets)

# ----------------------------------------------------------------#
# Create Token, Retrieve Spotify Artist
# ----------------------------------------------------------------#

artist_name = 'Jamie xx'
num_recommendations = 10
artist_stats = Spotify_Class.artist(artist_name)
client = SpotifyClientCredentials(client_id = artist_stats.cid, client_secret = artist_stats.secret)
sp = spotipy.Spotify(client_credentials_manager = client)
artist_stats.Get_Artist()

class_rec, df_rec, artist_list, song_list, popularity_list, track_number_list, uri_list = song_recommendations(artist_stats, num_recommendations)
Spotify_Def.new_line()
Apriori(artist_list)
Spotify_Def.new_line()
Apriori(song_list)
Spotify_Def.new_line()