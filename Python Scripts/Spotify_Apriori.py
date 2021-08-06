
# ----------------------------------------------------------------#
# Import Modules and Client
# ----------------------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import spotipy
import Spotify_Class
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------------------------------------------------------#
# Print New Line
# ----------------------------------------------------------------#

def new_line(): print('\n')

# ----------------------------------------------------------------#
# Retrieve recommended songs based off artist
# ----------------------------------------------------------------#

def Song_Generator(artist_ID, rclm):
    song_list = []
    artist_name, song_name, popularity, track_number, uri = [], [], [], [], []
    songs = sp.recommendations(seed_artists = [artist_ID], limit = rclm)
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

def song_recommendations(profile, loop, rclm):
    rec_compilation, df_rec_compilation = [], []
    artist_data, song_name_data, popularity_data, track_numbers_data, uris_data = [], [], [], [], []
    for k in range(0, loop):
        song_list, artist_names, song_names, popularity, track_numbers, uris = Song_Generator(profile.id, rclm)
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

def Apriori(data_set, mnspp):
    te = TransactionEncoder()
    te_ary = te.fit(data_set).transform(data_set)
    df = pd.DataFrame(te_ary, columns = te.columns_)
    item_sets = apriori(df, min_support = mnspp, use_colnames = True)
    return item_sets

# ----------------------------------------------------------------#
# Fetch Song from Spotify API
# ----------------------------------------------------------------#

def Get_Song(track):
    song_stats = Spotify_Class.song()
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

# ----------------------------------------------------------------#
# Object to List Conversion
# ----------------------------------------------------------------#

def obj_to_list(obj):
    new_list = []
    for k in range(0, len(obj)):
        item = list(obj[k])
        new_list.append(item[0])
    return new_list

# ----------------------------------------------------------------#
# Select Artist as Antecedent in Rules Analysis
# ----------------------------------------------------------------#

def artist_antecedent(df, selection):
    artist_rules = df[df['antecedents'] == {selection}]
    artist_rules = artist_rules[artist_rules['consequents'].apply(lambda x: len(x) < 2)]
    artist_rules = artist_rules.drop(['antecedent support','consequent support'], axis = 1)
    artist_rules = artist_rules.sort_values(by = 'support', ascending = False)
    return artist_rules

# ----------------------------------------------------------------#
# Print Custom Message
# ----------------------------------------------------------------#

def printer(set, message):
    print(message + '\n')
    for i in range(0, len(set)):
        temp = i+1
        print(str(temp) + '.' + '\t' + str(set[i]))

# ----------------------------------------------------------------#
# Print Custom Plot
# ----------------------------------------------------------------#

def plot_scatter(rule_set, x, y):
    fig, ax = plt.subplots(figsize = (12,8), dpi = 80)
    sns.stripplot(x = rule_set[x].round(3), y = rule_set[y], jitter = 0.25, size = 8, ax = ax, linewidth = 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    plt.title(x + ' vs ' + y, fontsize = 16)
    plt.show()

# ----------------------------------------------------------------#
# Format Recommended Songs List
# ----------------------------------------------------------------#

def create_song_list(set):
    new_list = []
    for j in range(0, len(set)):
        track = Get_Song(set[j])
        entry = set[j] + ' [' + track.artist_name + ']'
        new_list.append(entry)
    return new_list

# ----------------------------------------------------------------#
# Create Token, Retrieve Spotify Artist
# ----------------------------------------------------------------#

mode = 1    # 1 for artists, 2 for songs
all_artists_rules_print = False
artist_rules_print = False
artist_rules_plot = False
artist_all_results = True
song_all_results = False

artist_name = 'Dua Lipa'
artist_stats = Spotify_Class.artist(artist_name)
client = SpotifyClientCredentials(client_id = artist_stats.cid, client_secret = artist_stats.secret)
sp = spotipy.Spotify(client_credentials_manager = client)
artist_stats.Get_Artist()

if mode == 1:
    num_recommendations, min_supp, rec_limit  = 13, 0.3, 11 # ARTISTS -> 10 recs, limit 10, min_supp of 0.2
    class_rec, df_rec, artist_list, song_list, popularity_list, track_number_list, uri_list = song_recommendations(artist_stats, num_recommendations, rec_limit)
    frequent_itemsets_artist = Apriori(artist_list, min_supp)
    rules_for_artists = association_rules(frequent_itemsets_artist, metric = 'lift', min_threshold = 1)
    new_rules_artists = artist_antecedent(rules_for_artists, artist_name)
    if all_artists_rules_print:
        top_rules = rules_for_artists.sort_values(by='lift', ascending = False)[:50]
        print(top_rules)
    all_artists = list(new_rules_artists.loc[:,'consequents'])
    if artist_rules_print:
        print(rules_for_artists)
    if artist_rules_plot:
        plot_scatter(rules_for_artists, 'support', 'lift')
        plot_scatter(rules_for_artists, 'support', 'confidence')
        plot_scatter(rules_for_artists, 'lift', 'confidence')
        plot_scatter(rules_for_artists, 'leverage', 'conviction')
    if artist_all_results:
        new_line()
        print(new_rules_artists)
        artist_set = obj_to_list(all_artists)
        recommended_artists = artist_set[:5]
        artists_of_interest = artist_set[5:]
        new_line()
        printer(recommended_artists, 'Artists recommended for you:')
        new_line()
        printer(artists_of_interest, 'Other artists you may be interested in:')
        new_line()
elif mode == 2:
    num_recommendations, min_supp, rec_limit  = 4, 0.2, 6 # SONGS -> 4 recs, limit 6, min_supp 0.2
    class_rec, df_rec, artist_list, song_list, popularity_list, track_number_list, uri_list = song_recommendations(artist_stats, num_recommendations, rec_limit)
    frequent_itemsets_song = Apriori(song_list, min_supp)
    all_songs = frequent_itemsets_song.loc[0:20,'itemsets']
    song_set = obj_to_list(all_songs)
    songs = create_song_list(song_set)
    if song_all_results:
        new_line()
        printer(songs, 'Songs recommended for you:')
        new_line()



