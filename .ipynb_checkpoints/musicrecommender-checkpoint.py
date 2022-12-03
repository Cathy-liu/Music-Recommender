import streamlit as st
import pandas as pd
import numpy as np
pip3 install spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

st.title('Music Recommender')
import os
import numpy as np
import pandas as pd


from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

data = pd.read_csv('./songs datasets/data.csv')
#st.text('Spotify dataset loaded.')
number_cols=data.select_dtypes(np.number).columns
selected_cols=[elem for elem in number_cols if elem not in ('year','popularity','explicit')]
X=data[selected_cols]
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=9, 
                                   verbose=False))
                                 ], verbose=False)

#if st.checkbox('Show raw data'):
#    st.subheader('Raw data')
#    st.write(data)


st.markdown('Enter a Spotify playlist URL and get song recommendations based on the playlist.')
playlist_link = st.text_input("Your playlist URL/URI", key="playlist_link")
st.markdown("***")
st.markdown('Enter the full name of a song you like and the artists and get recommendations.')
name=st.text_input('The name of the song',key='name')
artists=st.text_input('The artists (optional)',key='artists')

if not (playlist_link or name):
    st.stop()
    
msg = 'Thank you for inputting a playlist, getting your recommendations now...'

        

if playlist_link:
    with st.spinner(msg):


#create Client ID and Client Secret on the 'Spotify for Developers' API 
        myid = 'c75d149c1c14479d8377cbed757567bb'
        mysecret='8c7aa3b6e08b4d609638b6f36a6c773b'
        os.environ['SPOTIFY_CLIENT_ID'] = str(myid)
        os.environ['SPOTIFY_CLIENT_SECRET']=str(mysecret)
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))
        
        playlist_URI = playlist_link.split("/")[-1].split("?")[0]
        track_uris = [x["track"]["uri"].split('track:')[-1] for x in sp.playlist_tracks(playlist_URI)["items"]]
        #st.write(track_uris)
        #st.write(sp.audio_features(track_uris[0])[0])
        
        def find_song_by_name(name, artists):
            song_data = defaultdict()
            #results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1) # api returns a dictionary contains info of the track; among them sub-key'items' contain primary artist info and also associate artists in 'external_urls'
            results=sp.search(q= 'track: {} artists: {}'.format(name,artists), limit=1)
            if results['tracks']['items'] == []:
                return None
            results = results['tracks']['items'][0]
            track_id = results['id'] 
            audio_features = sp.audio_features(track_id)[0]
            song_data['name'] = [name]
            #song_data['year'] = [year]
            song_data['explicit'] = [int(results['explicit'])]
            song_data['duration_ms'] = [results['duration_ms']]
            song_data['popularity'] = [results['popularity']]
            for key, value in audio_features.items():
                song_data[key] = value
            return pd.DataFrame(song_data)

        def get_song_data(song, spotify_data):
            try:
                song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                        & (spotify_data['artists'] == str(song['artists']))].iloc[0]
                return song_data
            except IndexError:
                return find_song_by_name(song['name'], song['artists'])
        
        def get_mean_vector(song_list, spotify_data):
    
            song_vectors = []

            for song in song_list:
                song_data = get_song_data(song, spotify_data)
                #print(song_data)
                if song_data is None:
                    print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
                    continue
                song_vector = song_data[selected_cols].values
                song_vectors.append(song_vector)  

            song_matrix = np.array(list(song_vectors))
            return np.mean(song_matrix, axis=0) #take mean for all the rows
        
        def flatten_dict_list(dict_list): # garther multiple 'name's into one list with the key 'name, and garther multiple 'artists' into one list with the key 'artists'
    
            flattened_dict = defaultdict() # use defaultdict to handle missing keys 
            for key in dict_list[0].keys():
                flattened_dict[key] = []

            for dictionary in dict_list:
                for key, value in dictionary.items():
                    flattened_dict[key].append(value)

            return flattened_dict
        
        
        def recommend_songs_from_text_input(song_list, spotify_data, n_songs=10):
    
            metadata_cols = ['name', 'year', 'artists']
            song_dict = flatten_dict_list(song_list)
            #print(song_dict)

            song_center = get_mean_vector(song_list, spotify_data)
            #print(song_center)
            scaler = song_cluster_pipeline.steps[0][1]
            scaled_data = scaler.fit_transform(spotify_data[selected_cols])
            #print('scaled_data[0]',scaled_data[0])
            scaled_song_center = scaler.transform(song_center.reshape(1, -1)) #reshape song_center from 1D array of 15 elements to 2D array (1 row 15 columns)
            #print(scaled_song_center)
            distances = cdist(scaled_song_center, scaled_data, 'cosine')
            #print('distances',len(distances[0])) # len(distances[0]) is about 17k+, means distances is 1 x 17k
            index = list(np.argsort(distances)[:, :n_songs][0])
            #print('list',np.argsort(distances)[:,:5]) # reflect the data structure here: scaled data 17k x 15, scaled song center 1 x 15, distances 1 x 17k, np.argsort(distances[0]) works -- np.argsort
            #print('index',index)

            rec_songs = spotify_data.iloc[index]
            #print('rec_songs',rec_songs)
            rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

            dict_singular=rec_songs[metadata_cols].to_dict(orient='records')
            #dict_singular=rec_songs[metadata_cols]
            for i in range(len(dict_singular)):
                name=dict_singular[i]['name']
                artists=dict_singular[i]['artists']
                dict_singular[i]['link']=sp.search(q= 'track: {} artists: {}'.format(name,artists), limit=1)['tracks']['items'][0]['album']['external_urls']['spotify']
                try:
                    dict_singular[i]['cover']=sp.search(q= 'track: {} artists: {}'.format(name,artists), limit=1)['tracks']['items'][0]['album']['images'][1]['url']
                except:
                    #dict_singular[i]['link']='-'
                    dict_singular[i]['cover']='-'

            return dict_singular,scaled_song_center
        
        
        def get_song_list_from_playlist(track_uris,spotify_data):
            song_list=[]
            for i in range(len(track_uris)):
                song_summary={}
                trackid=track_uris[i]
                try:
                    name=list(spotify_data[spotify_data['id']==trackid]['name'])[0]
                    artists=list(data[data['id']==track_uris[i]]['artists'])[0][1:-1].replace("'","").split(',')
                    song_summary['name']=name
                    song_summary['artists']=artists
                    song_list.append(song_summary)
                except:
                    pass
            return song_list
        
        
        def recommend_songs_from_playlist_url(track_uris, spotify_data):
            song_list=get_song_list_from_playlist(track_uris,data)
            return recommend_songs_from_text_input(song_list,spotify_data,n_songs=10)
        
        
        
        import webbrowser
        import matplotlib.pyplot as plt
        
        res,vis=recommend_songs_from_playlist_url(track_uris,data)
        
        col=data[selected_cols].columns
        vis_plt=pd.DataFrame(np.array([vis[0]]),columns=col)
        cols=['valence','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo']
        st.header('The general characteristics of this playlist')
        st.bar_chart(vis_plt[cols].T)
           
        st.header('Recommended songs for you')
        for i in range(len(res)):
            col1, col2= st.columns(2)
            with col1:
                image=res[i]['cover']
                try:
                    st.image(image)
                except:
                    pass

            with col2:
                st.write(res[i])
                url=res[i]['link']
                if st.button('Click to listen to Song {}'.format(i)):
                    webbrowser.open_new_tab(url)
            
          #  st.markdown("![Alt Text](https://i.scdn.co/image/ab67616d00001e024b292ed7c7360a04d3d6b74a)")
        


#if not name:
#    st.stop()


msg1 = 'Thank you for inputting the song name, getting your recommendations now...'
    
if name:
    with st.spinner(msg1):

#create Client ID and Client Secret on the 'Spotify for Developers' API 
        song_list=[{'name':name,'artists':[artists]}]
        myid = 'c75d149c1c14479d8377cbed757567bb'
        mysecret='8c7aa3b6e08b4d609638b6f36a6c773b'
        os.environ['SPOTIFY_CLIENT_ID'] = str(myid)
        os.environ['SPOTIFY_CLIENT_SECRET']=str(mysecret)
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))
        
        def find_song_by_name(name, artists):
            song_data = defaultdict()
            #results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1) # api returns a dictionary contains info of the track; among them sub-key'items' contain primary artist info and also associate artists in 'external_urls'
            results=sp.search(q= 'track: {} artists: {}'.format(name,artists), limit=1)
            if results['tracks']['items'] == []:
                return None
            results = results['tracks']['items'][0]
            track_id = results['id'] 
            audio_features = sp.audio_features(track_id)[0]
            song_data['name'] = [name]
            #song_data['year'] = [year]
            song_data['explicit'] = [int(results['explicit'])]
            song_data['duration_ms'] = [results['duration_ms']]
            song_data['popularity'] = [results['popularity']]
            for key, value in audio_features.items():
                song_data[key] = value
            return pd.DataFrame(song_data)

        def get_song_data(song, spotify_data):
            try:
                song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                        & (spotify_data['artists'] == str(song['artists']))].iloc[0]
                return song_data
            except IndexError:
                return find_song_by_name(song['name'], song['artists'])
        
        def get_mean_vector(song_list, spotify_data):
    
            song_vectors = []

            for song in song_list:
                song_data = get_song_data(song, spotify_data)
                #print(song_data)
                if song_data is None:
                    print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
                    continue
                song_vector = song_data[selected_cols].values
                song_vectors.append(song_vector)  

            song_matrix = np.array(list(song_vectors))
            return np.mean(song_matrix, axis=0) #take mean for all the rows
        
        def flatten_dict_list(dict_list): # garther multiple 'name's into one list with the key 'name, and garther multiple 'artists' into one list with the key 'artists'
    
            flattened_dict = defaultdict() # use defaultdict to handle missing keys 
            for key in dict_list[0].keys():
                flattened_dict[key] = []

            for dictionary in dict_list:
                for key, value in dictionary.items():
                    flattened_dict[key].append(value)

            return flattened_dict
        
        
        def recommend_songs_from_text_input(song_list, spotify_data, n_songs=10):
    
            metadata_cols = ['name', 'year', 'artists']
            song_dict = flatten_dict_list(song_list)
            #print(song_dict)

            song_center = get_mean_vector(song_list, spotify_data)
            #print(song_center)
            scaler = song_cluster_pipeline.steps[0][1]
            scaled_data = scaler.fit_transform(spotify_data[selected_cols])
            #print('scaled_data[0]',scaled_data[0])
            scaled_song_center = scaler.transform(song_center.reshape(1, -1)) #reshape song_center from 1D array of 15 elements to 2D array (1 row 15 columns)
            #print(scaled_song_center)
            distances = cdist(scaled_song_center, scaled_data, 'cosine')
            #print('distances',len(distances[0])) # len(distances[0]) is about 17k+, means distances is 1 x 17k
            index = list(np.argsort(distances)[:, :n_songs][0])
            #print('list',np.argsort(distances)[:,:5]) # reflect the data structure here: scaled data 17k x 15, scaled song center 1 x 15, distances 1 x 17k, np.argsort(distances[0]) works -- np.argsort
            #print('index',index)

            rec_songs = spotify_data.iloc[index]
            #print('rec_songs',rec_songs)
            rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

            dict_singular=rec_songs[metadata_cols].to_dict(orient='records')
            #dict_singular=rec_songs[metadata_cols]
            for i in range(len(dict_singular)):
                name=dict_singular[i]['name']
                artists=dict_singular[i]['artists']
                dict_singular[i]['link']=sp.search(q= 'track: {} artists: {}'.format(name,artists), limit=1)['tracks']['items'][0]['album']['external_urls']['spotify']
                try:
                    dict_singular[i]['cover']=sp.search(q= 'track: {} artists: {}'.format(name,artists), limit=1)['tracks']['items'][0]['album']['images'][1]['url']
                except:
                    #dict_singular[i]['link']='-'
                    dict_singular[i]['cover']='-'

            return dict_singular,scaled_song_center
 
        #st.write(recommend_songs_from_text_input(song_list, data, n_songs=10))

        import webbrowser
        import matplotlib.pyplot as plt
        
        res,vis=recommend_songs_from_text_input(song_list,data)
        
        col=data[selected_cols].columns
        vis_plt=pd.DataFrame(np.array([vis[0]]),columns=col)
        cols=['valence','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo']
        st.header('The general characteristics of this playlist')
        st.bar_chart(vis_plt[cols].T)
           
        st.header('Recommended songs for you')
        for i in range(len(res)):
            col1, col2= st.columns(2)
            with col1:
                image=res[i]['cover']
                try:
                    st.image(image)
                except:
                    pass

            with col2:
                st.write(res[i])
                url=res[i]['link']
                if st.button('Click to listen to Song {}'.format(i)):
                    webbrowser.open_new_tab(url)