
# ----------------------------------------------------------------#
# Import Modules and Client
# ----------------------------------------------------------------#

import pandas as pd
import spotipy
import Spotify_Class
from spotipy.oauth2 import SpotifyClientCredentials
import time

# ----------------------------------------------------------------#
# Export Dataframe to Microsoft Excel
# ----------------------------------------------------------------#

def export_file(df_conv, filename, extension):
    outputFile = filename + '.' + extension
    writer = pd.ExcelWriter(outputFile, engine = 'xlsxwriter')
    df_conv.to_excel(writer, 'Top 10')
    workbook = writer.book
    worksheet = writer.sheets["Top 10"]
    worksheet.set_column('A:Z', 18)
    writer.save()

# ----------------------------------------------------------------#
# Read CSV File into DataFrame
# ----------------------------------------------------------------#

def read_file(csv_file):
    return pd.read_csv(csv_file)

# ----------------------------------------------------------------#
# Print New Line
# ----------------------------------------------------------------#

def new_line(): print('\n')

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
# Classify Song's Success (Massive Hit, Hit, Minor Hit, dud)
# ----------------------------------------------------------------#

def hit_or_miss(pop_score):
    if pop_score >= 50: return 1
    else: return 0

# ----------------------------------------------------------------#
# Quantify Explicit (1 = explict, 0 = not explicit)
# ----------------------------------------------------------------#

def explicit(song_explicit):
    if song_explicit == True: return 1
    elif song_explicit == False: return 0

# ----------------------------------------------------------------#
# Quantify Local (1 = local, 0 = not local)
# ----------------------------------------------------------------#

def local(song_local):
    if song_local == True: return 1
    elif song_local == False: return 0

# ----------------------------------------------------------------#
# Spotify API Artist Top Tracks
# ----------------------------------------------------------------#

def popular_songs(profile):
    top_tracks_basket = []
    artist_name, artist_genre, track_duration, track_explicit, track_name, track_popularity, hit = [], [], [], [], [], [], []
    all_top_tracks = sp.artist_top_tracks(profile.uri)
    top_tracks = all_top_tracks['tracks']
    if len(top_tracks) > 0:
        for k in range(0, len(top_tracks)):
            track = Spotify_Class.top_track()
            track.disc_number = top_tracks[k]['disc_number']
            track.duration_ms = top_tracks[k]['duration_ms']
            artist_name.append(artist_stats.spotify_name)
            artist_genre.append(artist_stats.genres)
            track_duration.append(top_tracks[k]['duration_ms'])
            track.explicit = top_tracks[k]['explicit']
            track_explicit.append(explicit(top_tracks[k]['explicit']))
            track.external_urls = top_tracks[k]['external_urls']
            track.href = top_tracks[k]['href']
            track.is_local = top_tracks[k]['is_local']
            track.is_playable = top_tracks[k]['is_playable']
            track.name = top_tracks[k]['name']
            track_name.append(top_tracks[k]['name'])
            track.popularity = top_tracks[k]['popularity']
            track_popularity.append(top_tracks[k]['popularity'])
            hit.append(hit_or_miss(int(top_tracks[k]['popularity'])))
            track.track_number = top_tracks[k]['track_number']
            track.uri = top_tracks[k]['uri']
            top_tracks_basket.append(track)
        acousticness, danceability, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence, duration_ms  = [], [], [], [], [], [], [], [], [], [], [], [], []
        for k in range(0, len(top_tracks_basket)):
            top_track_acoustics = Song_Features(top_tracks_basket[k].uri)
            acousticness.append(top_track_acoustics.acousticness)
            danceability.append(top_track_acoustics.danceability)
            energy.append(top_track_acoustics.energy)
            instrumentalness.append(float(top_track_acoustics.instrumentalness))
            key.append(top_track_acoustics.key)
            liveness.append(top_track_acoustics.liveness)
            loudness.append(top_track_acoustics.loudness)
            mode.append(top_track_acoustics.mode)
            speechiness.append(top_track_acoustics.speechiness)
            tempo.append(top_track_acoustics.tempo)
            time_signature.append(top_track_acoustics.time_signature)
            valence.append(top_track_acoustics.valence)
            duration_ms.append(top_track_acoustics.duration_ms)
        df = pd.DataFrame({'Artist':artist_name, 'Artist Genre':artist_genre, 'Song':track_name, 'Popularity':track_popularity, 'Hit?':hit, 'Duration':track_duration, 'Explicit':track_explicit, 'Acousticness':acousticness, 'Danceability':danceability, 'Energy':energy, 'Instrumentalness':instrumentalness, 'Key':key, 'Liveness':liveness, 'Loudness':loudness, 'Mode':mode, 'Speechiness':speechiness, 'Tempo':tempo, 'Time_Signature':time_signature, 'Valence':valence})  
        return df

# ----------------------------------------------------------------#
# Create Training Data Set
# ----------------------------------------------------------------#

frames = []
genre_artists = read_file('genre_artists.csv')

new_line()
for k in range(0, len(genre_artists)):
    print('Capture top 10 songs for artist:\t\t' + str(genre_artists.loc[k,'top100']))
    artist_stats = Spotify_Class.artist(str(genre_artists.loc[k,'top100']))
    if not artist_stats:
        continue
    client = SpotifyClientCredentials(client_id = artist_stats.cid, client_secret = artist_stats.secret)
    sp = spotipy.Spotify(client_credentials_manager = client)
    artist_stats.Get_Artist()
    df_top10 = popular_songs(artist_stats)
    frames.append(df_top10)
    time.sleep(1)

new_line()
result = pd.concat(frames)
export_file(result, 'top100', 'xlsx')
