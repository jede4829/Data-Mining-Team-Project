
# ----------------------------------------------------------------#
# Import Modules
# ----------------------------------------------------------------#

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------------------------------------------------------#
# Artist Class
# ----------------------------------------------------------------#

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
        self.cid = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'                       # Client ID
        self.secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'                    # Secret

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
        client = SpotifyClientCredentials(client_id = self.cid, client_secret = self.secret)
        sp = spotipy.Spotify(client_credentials_manager = client)
        results = sp.search(q='artist:' + self.input_name, type='artist')
        List_of_Artists = results['artists']
        Attributes_Artists = List_of_Artists['items']
        Top_Artist = Attributes_Artists[0]
        self.Artist_Profile(Top_Artist)

# ----------------------------------------------------------------#
# Song Class
# ----------------------------------------------------------------#

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

# ----------------------------------------------------------------#
# Audio Analysis Class
# ----------------------------------------------------------------#

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

# ----------------------------------------------------------------#
# Top Track Class
# ----------------------------------------------------------------#

class top_track:
    def __init__(self):
        self.disc_number = 0
        self.duration_ms = 0
        self.explicit = ''
        self.external_urls = ''
        self.href = ''
        self.is_local = ''
        self.is_playable = ''
        self.name = ''
        self.popularity = 0
        self.track_number = ''
        self.uri = ''


