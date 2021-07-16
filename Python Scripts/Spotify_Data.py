
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
# Spotify API Artist Top Tracks
# ----------------------------------------------------------------#

def popular_songs(profile):
    top_tracks_basket = []
    track_duration, track_explicit, track_is_local, track_name, track_popularity = [], [], [], [], []
    all_top_tracks = sp.artist_top_tracks(profile.uri)
    top_tracks = all_top_tracks['tracks']
    if len(top_tracks) > 0:
        for k in range(0, len(top_tracks)):
            track = Spotify_Class.top_track()
            track.disc_number = top_tracks[k]['disc_number']
            track.duration_ms = top_tracks[k]['duration_ms']
            track_duration.append(top_tracks[k]['duration_ms'])
            track.explicit = top_tracks[k]['explicit']
            track_explicit.append(top_tracks[k]['explicit'])
            track.external_urls = top_tracks[k]['external_urls']
            track.href = top_tracks[k]['href']
            track.is_local = top_tracks[k]['is_local']
            track_is_local.append(top_tracks[k]['is_local'])
            track.is_playable = top_tracks[k]['is_playable']
            track.name = top_tracks[k]['name']
            track_name.append(top_tracks[k]['name'])
            track.popularity = top_tracks[k]['popularity']
            track_popularity.append(top_tracks[k]['popularity'])
            track.track_number = top_tracks[k]['track_number']
            track.uri = top_tracks[k]['uri']
            top_tracks_basket.append(track)
        acousticness, danceability, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence, duration_ms  = [], [], [], [], [], [], [], [], [], [], [], [], []
        for k in range(0, len(top_tracks_basket)):
            top_track_acoustics = Song_Features(top_tracks_basket[k].uri)
            acousticness.append(top_track_acoustics.acousticness)
            danceability.append(top_track_acoustics.danceability)
            energy.append(top_track_acoustics.energy)
            instrumentalness.append(top_track_acoustics.instrumentalness)
            key.append(top_track_acoustics.key)
            liveness.append(top_track_acoustics.liveness)
            loudness.append(top_track_acoustics.loudness)
            mode.append(top_track_acoustics.mode)
            speechiness.append(top_track_acoustics.speechiness)
            tempo.append(top_track_acoustics.tempo)
            time_signature.append(top_track_acoustics.time_signature)
            valence.append(top_track_acoustics.valence)
            duration_ms.append(top_track_acoustics.duration_ms)
        df = pd.DataFrame({'Song':track_name, 'Popularity':track_popularity, 'Duration':track_duration, 'Explicit':track_explicit, 'Local':track_is_local, 'Acousticness':acousticness, 'Danceability':danceability, 'Energy':energy, 'Instrumentalness':instrumentalness, 'Key':key, 'Liveness':liveness, 'Loudness':loudness, 'Mode':mode, 'Speechiness':speechiness, 'Tempo':tempo, 'Time_Signature':time_signature, 'Valence':valence, 'Duration_ms':duration_ms})  
        return df

# ----------------------------------------------------------------#
# Create Token, Retrieve Spotify Artist, Capture Top 10
# ----------------------------------------------------------------#

artist_name = 'Metallica'
artist_stats = Spotify_Class.artist(artist_name)
client = SpotifyClientCredentials(client_id = artist_stats.cid, client_secret = artist_stats.secret)
sp = spotipy.Spotify(client_credentials_manager = client)
artist_stats.Get_Artist()
df_top10 = popular_songs(artist_stats)

Spotify_Def.new_line()
print('Artist Name:     ' + artist_stats.spotify_name)
print('Spotify URI:     ' + artist_stats.uri)
print('Followers:       ' + str(artist_stats.followers))
print('Genre List:      ' + str(artist_stats.genres))
print('Popularity:      ' + str(artist_stats.popularity))
Spotify_Def.new_line()
print(df_top10)
Spotify_Def.new_line()
