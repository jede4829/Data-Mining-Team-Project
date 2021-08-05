from tracks_header import *

def probability(x, mean, stdev):
    exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def separate(df):
    df['popularity'].values[df['popularity'].values < 50] = 0
    df['popularity'].values[df['popularity'].values >= 50] = 1
    print("popular: %d" % len(df.loc[df['popularity']==1]))
    return df

def scatterplot(df, attr1, attr2):
    df3 = df.loc[df['popularity']==0]
    df3 = sample_size(df3, 1000)
    plt.scatter(df3[attr1], df3[attr2], c='lightblue')
    df2 = df.loc[df['popularity']==1]
    #df2 = sample_size(df2, 6000)
    print(len(df2))
    plt.scatter(df2[attr1], df2[attr2], c='coral')
    plt.title(attr1 + ' vs ' + attr2)
    plt.xlabel(attr1)
    plt.ylabel(attr2)
    plt.show()

def summary(df):
    dct = {}
    lst = [[] for i in range(2)]
    count = 0
    df2 = df.loc[df['popularity']==1]
    #df2 = sample_size(df2, 1000)
    df3 = df.loc[df['popularity']==0]
    df3 = sample_size(df3, 1000)
    for i in ['loudness','danceability', 'explicit']:
        lst[0].append([df3[i].mean(), df3[i].std(), len(df3[i])])
        dct[0]=lst[0]
    for i in ['loudness','danceability', 'explicit']:
        lst[1].append([df2[i].mean(), df2[i].std(), len(df2[i])])
        dct[1]=lst[1]
    return dct

def class_prob(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= probability(row[i], mean, stdev)
    return probabilities

def radar_bayes(dataset, df, lst_pop):
    labels = list(dataset)[1:]
    labels = [*labels,labels[0]]
    unpop = df.loc[(df['popularity']<=100)&(df['release_date']<=2021)].drop('release_date',1).mean().tolist()[1:]
    popop = df.loc[(df['popularity']>=80)&(df['release_date']>=2020)].drop('release_date',1).mean().tolist()[1:]
    pop = dataset.loc[dataset['popularity']==1].mean().tolist()[1:]
    lst_pop = [*lst_pop, lst_pop[0]]
    pop = [*pop, pop[0]]
    unpop = [*unpop, unpop[0]]
    popop = [*popop, popop[0]]
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(lst_pop))
    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    plt.plot(label_loc, lst_pop, label='Predicted Popular Class')
    plt.fill(label_loc, lst_pop, alpha=0.25, facecolor='blue')
    
    plt.plot(label_loc, unpop, label='All Songs')
    plt.fill(label_loc, unpop, alpha=0.25, facecolor='orange')
    plt.title('Naive Bayes Popularity Prediction', size=20, y=1.05)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=labels)
    plt.legend()
    plt.show()

def predict_class(dataset, df):
    lst = ['explicit', 'danceability', 'energy', 'loudness', 'tempo']
    x = dataset.iloc[:,1:6].values
    y = dataset.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 5)
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy: %f" % accuracy_score(y_test, y_pred))
    print("Precision: %f" % precision_score(y_test, y_pred))
    print(cm)
    count=0
    obj_1=[0 for x in range(len(lst))]
    lst_pop = []
    for x in range(10000):
        attrlst=[]
        attrlst.append(random.randint(0,1))
        attrlst.append(round(random.uniform(0,1), 2))
        attrlst.append(round(random.uniform(0,1),2))
        attrlst.append(round(random.uniform(0,1), 2))
        attrlst.append(round(random.uniform(0,1),2))
        y_pred = classifier.predict([attrlst])
        if y_pred == 1:
            count+=1
            for i in range(len(lst)):
                obj_1[i] += attrlst[i]
    for x in range(len(lst)):
        obj_1[x] = obj_1[x]/(count)
        lst_pop.append(obj_1[x])
    radar_bayes(dataset, df, lst_pop)

def radar(df):
    labels = list(df.drop('release_date',1))[1:]
    labels.append(labels[0])
    unpop = df.loc[(df['popularity']<=100)&(df['release_date']<=2021)].drop('release_date',1).mean().tolist()[1:]
    pop = df.loc[(df['popularity']>=80)&(df['release_date']<=2019)].drop('release_date',1).mean().tolist()[1:]
    pop20 = df.loc[(df['popularity']>=80)&(df['release_date']>=2020)].drop('release_date',1).mean().tolist()[1:]
    pop = [*pop, pop[0]]
    pop20 = [*pop20, pop20[0]]
    unpop = [*unpop, unpop[0]]
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(labels))
    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    plt.plot(label_loc, pop20, label='Popular 2020-2021')
    plt.fill(label_loc, pop20, alpha=0.25, facecolor='blue')
    plt.plot(label_loc, pop, label='Popular 1922-2019')
    plt.fill(label_loc, pop, alpha=0.25, facecolor='orange')
    plt.plot(label_loc, unpop, label='All Songs 1922-2021')
    plt.fill(label_loc, unpop, alpha=0.25, facecolor='green')
    plt.title('Mean Values', size=20, y=1.05)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=labels)
    plt.legend()
    plt.show()
    
def gaussian(dataset):
    df2 = dataset.loc[dataset['popularity']==1]
    #df2 = sample_size(df2,25000)
    df3 = dataset.loc[dataset['popularity']==0]
    #df3 = sample_size(df3,4000)
    frames = [df2,df3]
    dataset = pd.concat(frames)
    x = dataset.iloc[:,1:9].values
    y = dataset.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 5)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy: %f" % accuracy_score(y_test, y_pred))
    print("Precision: %f" % precision_score(y_test, y_pred))
    print(cm)

def main():
    normalize_expr = to_normalize
    normalize_expr.remove('release_date')
    normalize_expr.remove('popularity')
    df = read_file("tracks.csv")
    df = clean_dates(df)
    df = clean_data(df)
    df = normalize(df, normalize_expr)
    df_dates = df
    df_all = df
    #df = sample_size(df, 10000)
    #print(df['popularity'])
    df = df.loc[df['release_date']>=2020]
    df = df.loc[df['popularity']>=7]
    print("total: %d" % len(df))
    df = separate(df)
    df = df[[col for col in df if col in ['popularity','tempo','loudness','explicit','energy',
                                          'danceability']]]
    df_dates = df_dates[[col for col in df_dates if col in ['release_date','popularity','tempo',
                                                            'loudness','explicit','energy',
                                                            'danceability']]]
    df_all = df_all[[col for col in df_all if col in ['release_date','mode','valence',
                                                      'popularity','acousticness','tempo',
                                                      'loudness','explicit','instrumentalness',
                                                      'energy','danceability','speechiness']]]
    radar(df_all)
    predict_class(df,df_dates)
    gaussian(df)
    #scatterplot(df, 'loudness','danceability')
    #print(summary(df))
    #print(df.iloc[0])
    #print(class_prob(summary(df), [1,1,1]))

main()
