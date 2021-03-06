from tracks_header import *

#uses tracks.csv from kaggle. File is too large for github

def regr_stats(df):
    #performs multiple linear regression
    #returns r-squared and adj r-squared
    dep_var = df['popularity']
    ind_vars = df[['release_date', 'danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                   'valence', 'tempo', 'time_signature','duration_ms', 'explicit']]
    reg = linear_model.LinearRegression()
    reg.fit(ind_vars, dep_var)
    yhat = reg.predict(ind_vars)
    SS_Residual = sum((dep_var-yhat)**2)       
    SS_Total = sum((dep_var-np.mean(dep_var))**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(dep_var)-1)/(len(dep_var)-ind_vars.shape[1]-1)
    return r_squared, adjusted_r_squared

def get_all_cc(headings, file, output=True):
    #prints correlation coefficient of every heading with popularity
    cdct = {}
    for i in range(len(headings)):
        if headings[i] == "release_date" or headings[i] == "popularity":
            continue
        try:
            x = correlation(headings[i], 'popularity', file)
            if output:
                print("Correlation Coefficient between %s and popularity: %f"
                        % (headings[i], x))
            cdct[headings[i]] = x
        except:
            print("could not calculate correlation between popularity and %s"
                  % (headings[i]))
    return cdct

def get_cc_decade(headings, df):
    #prints attributes with largest correlation coefficients for each decade
    years = [1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2022]
    x, y, j = 0, 0, 0
    for i in range(len(years)-1):
        j = 0
        df2 = df.loc[((df['release_date']>=years[i]) & (df['release_date']<years[i+1]))]
        print('Largest Correlation Coefficients '+ str(years[i]) + "-" + str(years[i+1]-1) + ":")
        dct = get_all_cc(headings, df2, output=False)
        y = dict(sorted(dct.items(), key=lambda item: abs(item[1]), reverse=True))
        for k, v in y.items():
            if j < 4:
                print(k + ": " + str(v))
                j += 1
        print()    

def plot_scatter(df, attr1, attr2):
    #scatter plot of two attributes
    df.plot(x=attr1, y=attr2, style='o')  
    plt.title(attr1 + ' vs ' + attr2)  
    plt.xlabel(attr1)  
    plt.ylabel(attr2)  
    plt.show()

def plot_pop_decade(df):
    #histograms of popularity for each decade
    #broken up by decade to decrease release_date's contribution to popularity
    years = [1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2022]
    x, y = 0, 0
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), sharey=True)
    plt.tight_layout(pad=4.0)
    for i in range(len(years)-1):
        df2 = df.loc[((df['release_date']>=years[i]) & (df['release_date']<years[i+1]))]
        sns.histplot(df2['popularity'],ax=axes[x,y])
        axes[x,y].set_title(str(years[i]) + "-" + str(years[i+1]-1))
        median = df2['popularity'].median()
        axes[x,y].axvline(median, color='r', linestyle='--')
        if y == 4:
            y = 0
            x += 1
        else:
            y += 1
        
    fig.suptitle('Song Popularity Distribution By Decade')
    plt.show()

def plot_reg_decade(df, attribute):
    #linear regression of attribute and popularity by decade
    #vertical dashed line is median
    years = [1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2022]
    x, y = 0, 0
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), sharey=True)
    plt.tight_layout(pad=4.0)
    sns.set_theme(color_codes=True)
    for i in range(len(years)-1):
        df2 = df.loc[((df['release_date']>=years[i]) & (df['release_date']<years[i+1]))]
        df2 = df2.sample(n=1000)
        sns.regplot(x=df2[attribute], y=df2["popularity"], data=df2, line_kws={"color": "red"},ax=axes[x,y]);
        axes[x,y].set_title(str(years[i]) + "-" + str(years[i+1]-1))
        median = df2[attribute].median()
        axes[x,y].axvline(median, color='r', linestyle='--')
        if y == 4:
            y = 0
            x += 1
        else:
            y += 1
        
    fig.suptitle('Song Pop and '+ attribute +' By Decade')
    plt.show()

def plot_box_decade(df, attribute):
    #box plot of attribute's affect on popularity for each decade
    #two box plots per graph, split data at mean
    #if below mean, value = 0, 1 if above
    years = [1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2022]
    x, y = 0, 0
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), sharey=True)
    plt.tight_layout(pad=4.0)
    sns.set_theme(color_codes=True)
    for i in range(len(years)-1):
        df2 = df.loc[((df['release_date']>=years[i]) & (df['release_date']<years[i+1]))]
        mean = df2[attribute].mean()
        df2[attribute].values[df2[attribute].values > mean] = 1
        df2[attribute].values[df2[attribute].values <= mean] = 0
        sns.boxplot(x=df2[attribute], y=df2["popularity"], data=df2,ax=axes[x,y]);
        axes[x,y].set_title(str(years[i]) + "-" + str(years[i+1]-1))
        mean = df2[attribute].mean()
        axes[x,y].axvline(mean, color='r', linestyle='--')
        if y == 4:
            y = 0
            x += 1
        else:
            y += 1
        
    fig.suptitle('Song Popularity and '+ attribute +' By Decade')
    plt.show()

def lin_regr(df):
    #linear regression, date and popularity
    #can use lin_reg instead
    X = df['release_date'].values.reshape(-1,1)
    y = df['popularity'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_test, y_test)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    plt.scatter(X_test, y_test,  color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show()

def lin_reg(df, attribute):
    #linear regression between attribute and popularity, all years
    #slow, reduce size of dataframe (# of rows) before using
    sns.set_theme(color_codes=True)
    sns.regplot(x=df[attribute], y=df["popularity"], data=df, line_kws={"color": "red"});
    plt.show()

def boxplot(df, attribute):
    #boxplot of attribute and popularity
    #split at mean, 0 below, 1 above
    sns.set_theme(color_codes=True)
    mean = df[attribute].mean()
    df[attribute].values[df[attribute].values > mean] = 1
    df[attribute].values[df[attribute].values <= mean] = 0
    sns.boxplot(x=df[attribute], y=df["popularity"], data=df);
    plt.show()

def genres_apriori(dataset):
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df.to_csv('out.csv')
    itemsets = apriori(df, min_support=0.1,use_colnames=True)
    
    print(itemsets)
    
def main():
    #prep dataframe
    df = read_file("tracks.csv")
    dfa = read_file("data_by_artist_o.csv")
    df = clean_dates(df)
    df = fix_order(df)
    df = clean_data(df)
    headings = get_headings(df)[4:]
    dfa = dfa.loc[dfa['popularity']>= 50]
    dfa = dfa.loc[dfa['count']>= 30]
    dfa = dfa.loc[dfa['genres'] != '[]']
    dfa = clean_genres(dfa)
    genres_apriori(dfa['genres'])
    #sample usage of above functions for plotting, etc
    get_cc_decade(headings, df)
    plot_scatter(df, 'release_date','popularity')
    plot_pop_decade(df)
    plot_reg_decade(df,'loudness')
    plot_box_decade(df,'explicit')
    df2 = sample_size(df, 1000) #reduce dataframe to 1000 randomly selected rows
    lin_reg(df2, 'release_date')
    lin_reg(df2, 'energy')
    boxplot(df2, 'explicit')
    boxplot(df2, 'loudness')
    return
    
main()
