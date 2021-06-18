#!/usr/bin/python
import sys
import pandas as pd

def correlation(attribute1, attribute2, file):
    attr1 = file[attribute1]
    attr2 = file[attribute2]
    correlation = attr1.corr(attr2)
    return correlation

def read_file(fileName):
    file = pd.read_csv(fileName)
    return file

def get_headings(file):
    headings = list(file.columns)
    return headings

def find_all_cc(headings, file):
     for i in range(len(headings)):
         for j in range(i+1,len(headings)):
             x = correlation(headings[i], headings[j], file)
             print("Correlation Coefficient between %s and %s: %f"
                    % (headings[i], headings[j], x))

def get_artist(df):
    artist = input("Enter artist name to search for similar artists: ")
    if artist == 'n':
        return 0
    similar_artists(df, artist.lower())
    return 1

def similar_artists(df, artist):
    df["artists"] = df["artists"].str.lower()
    artistRow = df.loc[lambda df: df.artists == artist]
    if artistRow.empty:
        print("Artist not found, check spelling.")
    else:
        dic = {}
        i = 0
        songs = artistRow['count'].iloc[0]
        countdiff = songs/2
        popularity = artistRow['popularity'].iloc[0]
        popdiff = popularity*.85
        genre = artistRow.genres.iloc[0][1:-1].split(",")
        for x in range(len(genre)):
            genre[x] = genre[x].strip()[1:-1]
        for i in range(df.shape[0]):
            currow = df.iloc[i].to_frame().T
            sgenre = currow.genres.iloc[0][1:-1].split(",")
            for x in range(len(sgenre)):
                sgenre[x] = sgenre[x].strip()[1:-1]
            if any(item in genre for item in sgenre):
                if not currow.loc[([currow.popularity] >= popdiff)
                    & ([currow.popularity] <= artistRow.popularity.iloc[0]+10)].empty:
                    if not currow.loc[([currow.energy] >= artistRow.energy.iloc[0]-.2)
                        & ([currow.energy] <= artistRow.energy.iloc[0]+.2)].empty:
                        if not currow.loc[[currow['count']] >= countdiff].empty:
                            dic[currow['artists'].iloc[0]] = currow['popularity'].iloc[0]
                            if len(dic) >= 5:
                                break
        if artistRow.artists.iloc[0] in dic:
            dic.pop(artistRow.artists.iloc[0])
        for w in sorted(dic, key=dic.get, reverse=True):
            print(w)
        if len(dic) == 0:
            print("Similar artists not found.")

def main():
    if len(sys.argv) > 1:
        try:
            df = read_file(sys.argv[1])
        except:
            print("File not found: %s" % sys.argv[1])
            return
    else:
        df = read_file("data_by_artist_o.csv")
    headings = get_headings(df)[2:]
    find_all_cc(headings, df)
    print("----\nArtist Search ('n' to exit)\n----")
    notDone = get_artist(df)
    while(notDone):
        notDone = get_artist(df)

if __name__ == "__main__":
    main()
