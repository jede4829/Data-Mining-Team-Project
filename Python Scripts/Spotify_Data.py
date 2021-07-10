
# ----------------------------------------------------------------#
# Import Modules and Client
# ----------------------------------------------------------------#

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import spotipy
import Spotify_Class
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------------------------------------------------------#
# Create Token, Retrieve Spotify Artist
# ----------------------------------------------------------------#

artist_name = 'The Weeknd'
artist_stats = Spotify_Class.artist(artist_name)
client = SpotifyClientCredentials(client_id = artist_stats.cid, client_secret = artist_stats.secret)
sp = spotipy.Spotify(client_credentials_manager = client)
artist_stats.Get_Artist()

# ----------------------------------------------------------------#
# Retrieve recommended songs based off artist
# ----------------------------------------------------------------#

def Song_Generator(artist_ID):
    song_list = []
    artist_name, song_name, popularity, track_number, uri = [], [], [], [], []
    songs = sp.recommendations(seed_artists = [artist_ID])
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
# Retrieve Audio Analysis of Single
# ----------------------------------------------------------------#

def Song_Features(single):
    features = sp.audio_features(single)
    track_stats = Spotify_Class.track_features()
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

# ----------------------------------------------------------------#
# Spotify API Collect List of Song Recommendations
# ----------------------------------------------------------------#

def song_recommendations(profile):
    loop = 1
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
    return rec_compilation, df_rec_compilation

class_rec, df_rec = song_recommendations(artist_stats)

# ----------------------------------------------------------------#
# Spotify API Capture and Tabulate Song Acoustics
# ----------------------------------------------------------------#

def song_acoustics(rec_class_list):
    acoustics_class_basket, acoustics_df_basket = [], []
    acoustics_complete, artists, songs = [], [], []
    acousticness, danceability, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence, duration_ms  = [], [], [], [], [], [], [], [], [], [], [], [], []
    for j in range(0, len(rec_class_list)):
        single_rec_class = rec_class_list[j]
        for k in range(0, len(single_rec_class)):
            single = single_rec_class[k]
            artist = single.artist_name
            song = single.song_name
            uri = single.song_uri
            single_acoustics = Song_Features(uri)
            acoustics_complete.append(single_acoustics)
            artists.append(artist)
            songs.append(song)
            acousticness.append(single_acoustics.acousticness)
            danceability.append(single_acoustics.danceability)
            energy.append(single_acoustics.energy)
            instrumentalness.append(single_acoustics.instrumentalness)
            key.append(single_acoustics.key)
            liveness.append(single_acoustics.liveness)
            loudness.append(single_acoustics.loudness)
            mode.append(single_acoustics.mode)
            speechiness.append(single_acoustics.speechiness)
            tempo.append(single_acoustics.tempo)
            time_signature.append(single_acoustics.time_signature)
            valence.append(single_acoustics.valence)
            duration_ms.append(single_acoustics.duration_ms)
        df = pd.DataFrame({'Artist':artists, 'Song':songs, 'Acousticness':acousticness, 'Danceability':danceability, 'Energy':energy, 'Instrumentalness':instrumentalness, 'Key':key, 'Liveness':liveness, 'Loudness':loudness, 'Mode':mode, 'Speechiness':speechiness, 'Tempo':tempo, 'Time_Signature':time_signature, 'Valence':valence, 'Duration_ms':duration_ms})
        acoustics_class_basket.append(acoustics_complete)
        acoustics_df_basket.append(df)
        artists.clear()
        songs.clear()
        acousticness.clear()
        danceability.clear()
        energy.clear()
        instrumentalness.clear()
        key.clear()
        liveness.clear()
        loudness.clear()
        mode.clear()
        speechiness.clear()
        tempo.clear()
        time_signature.clear()
        valence.clear()
        duration_ms.clear()
    return acoustics_class_basket, acoustics_df_basket

test1, test2 = song_acoustics(class_rec)
print(test2[0])
# print(test2[0],'\n\n',test2[1])





