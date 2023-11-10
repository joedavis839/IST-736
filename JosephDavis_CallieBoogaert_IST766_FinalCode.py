#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 20:49:25 2023

@author: Callie Boogaert, Joseph Davis
"""
#One Very Long and consolidated python file
#Import all needed packages
import pandas as pd
import numpy as np
import requests
import json
import lyricsgenius
from requests.exceptions import Timeout
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, f1_score, classification_report)

#Import the top 500 songs according to Rollling Stones Magazine and
#The accompanying lyrics

RS500 = pd.read_csv("rollingstone_2021.csv")
Genres = pd.read_csv("Genres.csv")
RS500 = pd.merge(RS500, Genres, on=['Artist','Title'])
#RS500.head()

# Clean up df
#Rename the Unnamed column to rank
RS500=RS500.rename(columns={'Unnamed: 0':'Rank'})
#Re-do the rank so that it is 500-1
RS500['Rank'] = -1*(RS500['Rank']-500)
#Clean up some song titles/ artist names
RS500.replace('Move Your Body (The House Music Anthem)', 'Move Your Body', inplace = True)
RS500.replace('Pt. 1-Acknowledgement', 'A Love Supreme, Part 1:Acknowledgement', inplace = True)
RS500.replace('Booker T. and the MGs', "Booker T. & the M.G.'s", inplace = True)
RS500.replace('King Tubby Meets the Rockers Uptown', 'King Tubby Meets Rockers Uptown', inplace = True)
#RS500.head(500)

################################
#######Data Explorations########
################################
RS500.describe()

##Bin Variables to create labels for analysis
# Create variable 100s_rank and 50s_rank by binning rank
RS500['100s_rank'] = pd.cut(RS500['Rank'],[0,100,200,300,400,500], labels=['top 100', '100-200','200-300','300-400','400-500'])
RS500['50s_rank'] = pd.cut(RS500['Rank'], [0,50,100,150,200,250,300,350,400,450,500], labels = ['top 50', '50-100', '100-150'
                                                                                               ,'150-200', '200-250','250-300'
                                                                                               ,'300-350', '350-400', '400-450'
                                                                                               ,'450-500'])
# Create variable pop_grade by binning popularity
RS500['pop_grade'] = pd.cut(RS500['Popularity'], [0,59,69,79,89,100], labels = ['F', 'D', 'C', 'B', 'A'])

# Create a variable decade by binning years
#Important to not we will be refering to 2000 as 00s, 2010 as 10s and 2020 as 20s
RS500['decade'] = pd.cut(RS500['Year'], [1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,np.inf], 
                         labels = ['30s', '40s', '50s', '60s', '70s', '80s', '90s', '00s', '10s', '20s'])

#Check for Nan
RS500.isnull().sum()
RS500['pop_grade'] = RS500['pop_grade'].fillna('F')

## Rolling Stones Data Exploration

#Graph the most common artists
# artist Artists
artist_count = RS500['Artist'].value_counts()
top_artists = artist_count.head(10)
#Plot most common artists
plt.figure(figsize=(12,6))
top_artists.plot(kind='bar', color='b')
plt.title('Top 10 Artists with the Most Songs')
plt.xlabel('Artist')
plt.ylabel('Song Count')
plt.xticks(rotation = 45, ha = 'right')
plt.tight_layout()
plt.show()

## Graph the most common years
# song count
yr_count = RS500['Year'].value_counts()
top_yrs = yr_count.head(15)

#Plot most common years
plt.figure(figsize=(12,6))
top_yrs.plot(kind='bar', color='b')
plt.title('Top 15 Years with the Most Songs')
plt.xlabel('Year')
plt.ylabel('Song Count')
plt.xticks(rotation = 45, ha = 'right')
plt.tight_layout()
plt.show()

## Graph songs by decade
# song count
dec_count = RS500['decade'].value_counts()
#Plot most common decades
plt.figure(figsize=(12,6))
dec_count.plot(kind='bar', color='b')
plt.title('Songs by Decades')
plt.xlabel('Decade')
plt.ylabel('Song Count')
plt.xticks(rotation = 45, ha = 'right')
plt.tight_layout()
plt.show()

## Graph songs by genre
# song count
gen_count = RS500['Genre'].value_counts()
#Plot most common decades
plt.figure(figsize=(12,6))
gen_count.plot(kind='bar', color='b')
plt.title('Songs by Genre')
plt.xlabel('Genre')
plt.ylabel('Song Count')
plt.xticks(rotation = 45, ha = 'right')
plt.tight_layout()
plt.show()

##ScatterPlot of Popularity and year
plt.scatter(RS500['Year'], RS500['Popularity'])
plt.show

##ScatterPlot of danceability and year
plt.scatter(RS500['Year'], (RS500['danceability']*100))
plt.show

## Box plot of 100s_rank and popularity
cols = {'Rank by 100s': '100s_rank', 'Rank by 50s':'50s_rank',}
for key, col in cols.items():
    plt.figure(figsize = (20,8))
    sns.boxplot(data = RS500, x = col, y = 'Popularity')
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('Popularity')
    plt.title('Popularity by ' + key)
    plt.show




#############################
###Data Compiling/Dleaning###
#############################

#Create a dictionary for songs and artists
songs = pd.Series(RS500.Artist.values, index=RS500.Title).to_dict()
#print(songs)

#Create lists for the lyric, song, title, and artist
lyric=[]
title=[]
artist=[]

#Create a function to loop through dictionary and retrieve lyrics
def get_lyrics():
    api=lyricsgenius.Genius('JVGzCpUT-JmDRBF65eIW5_M4EEIjn901qDUQqfhPrBa3A69ddp8nVdDQgjg7GRSR')
    api.timeout = 15
    api.sleep = 45
    for key, value in songs.items():
        retries = 0
        while retries <3:
            try:
                song = api.search_song(key,value)
                lyric.append(song.lyrics)
                title.append(song.title)
                artist.append(song.artist)
                break
            except Timeout as e:
                retries +=1
                continue

#Following songs do not retrieve lyrics so drop them
del songs['So What']
del songs['Green Onions']
del songs['Cissy Strut']
#run the function created above
get_lyrics()
lyric_df = pd.DataFrame({'lyrics':lyric,'Title':title,'Artist': artist})
lyric_df.head(500)

#lyric list
lyric
title
artist
## Once lyric list is cleaned up can create dataframe
lyric_df = pd.DataFrame({'lyrics':lyric,'Title':title,'Artist': artist})

## Clean up the lyric dataframe
 #remove the [introduction] [outtro] things in brackets
lyric_df['lyrics'] = lyric_df['lyrics'].apply(lambda x: re.sub(r'\[.*?\]', ' ', x))
    #remove the /n at the end of lines
lyric_df['lyrics'] = lyric_df['lyrics'].apply(lambda x: re.sub(r'\n', '', x))
    #remove the beginning Contributors things
lyric_df['lyrics'] = lyric_df['lyrics'].apply(lambda x: re.sub(r'Contributors[A-Za-z\s]*', '', x))

# remove everything up until the word Lyrics
lyric_df['lyrics'] = lyric_df['lyrics'].apply(lambda x: re.sub(r'^.*?Lyrics', '', x))

#Remove all special characters and numbers
lyric_df['lyrics'] = lyric_df['lyrics'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]','', x))

# Insert spaces between words where needed
lyric_df['lyrics'] = lyric_df['lyrics'].apply(lambda x: re.sub(r'([a-z])([A-Z])',r'\1 \2', x))

#Remove numbers
lyric_df['lyrics'] = lyric_df['lyrics'].apply(lambda x: re.sub(r'\d+', '', x))

#remove Embed from the end of each string
lyric_df['lyrics']  = [s[:-len('Embed')].strip() if s.endswith('Embed') else s for s in lyric_df['lyrics']]

lyric_df['lyrics'][0]
#print(index)

############################
########word cloud##########
############################
##All Lyrics
all_lyric = " ".join(i for i in lyric_df.lyrics)
lyric_cloud = WordCloud(max_font_size=50, max_words=50, stopwords= set(STOPWORDS), background_color='white', colormap = 'winter' ).generate(all_lyric)
plt.figure()
plt.imshow(lyric_cloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()
##All song titles
titles = " ".join(i for i in lyric_df.Title)
title_cloud = WordCloud(max_font_size=50, max_words=50, stopwords= set(STOPWORDS), background_color='white', colormap = 'winter' ).generate(titles)
plt.figure()
plt.imshow(title_cloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


############################
########Final Cleaning######
############################
## Merge the lyrics into the RS500 dataframe

#Rename values in Artist so RS500 and lyric_df match
RS500['Artist'] = RS500['Artist'].replace(['Juvenile feat. Lil Wayne and Mannie Fresh','The Go-Gos',
                                           'Gladys Knight and the Pips', 'Missy Elliot','Migos feat. Lil Uzi Vert',
                                           "Megan Thee Stallion feat. Beyoncé",'Pete Rock and CL Smooth',
                                          'Queen and David Bowie', 'Sugar Hill Gang', 
                                           'Blackstreet feat. Dr. Dre and Queen Pen','Craig Mack feat. Notorious B.I.G.',
                                          'The Mamas and the Papas', 'Allman Brothers Band', 'Cat Stevens/Yusuf',
                                           'Rufus and Chaka Khan', 'J Balvin and Bad Bunny', 'Hall and Oates',
                                          'Rihanna feat. Jay-Z', 'Smokey Robinson and the Miracles',
                                          'The B-52', 'Beach Boys', 'Usher feat. Lil Jon and Ludacris',
                                          'Toots and the Maytals', 'Drake feat. Rihanna', 'Gil-Scott Heron',
                                          'Martha Reeves and the Vandellas', 'Stills and Nash', 'Outkast',
                                          'Bob Marley and the Wailers', 'Eric B. and Rakim',
                                           'Martha and the Vandellas', 'Drake feat. Majid Jordan', 'Rob Base and DJ E-Z Rock', 
                                           'UGK feat. Outkast',
                                          'Simon and Garfunkel', 'Earth, Wind, and Fire', 'Grandmaster Flash and the Furious Five',
                                          'Smokey Robinson and the Miracles', 'Notorious B.I.G.', 'Beyoncé feat. Jay-Z',
                                          'The Wailers', 'Prince and the Revolution','Guns N'],
                                        ['Juvenile', "The Go-Go's",'Gladys Knight & The Pips','Missy Eliott',
                                         'Migos','Megan Thee Stallion', 'Pete Rock & C.L. Smooth',
                                        'Queen & David Bowie', 'Sugarhill Gang', 
                                         'Blackstreet', 'Craig Mack', 
                                         'The Mamas & The Papas', 'The Allman Brothers Band', 'Cat Stevens',
                                         'Rufus','Bad Bunny & J Balvin','Daryl Hall & John Oates', 'Rihanna',
                                        'Smokey Robinson & The Miracles', 'The B-52s','The Beach Boys',
                                        'Usher', 'Toots & the Maytals', 'Drake', 'Gil Scott-Heron',
                                        'Martha Reeves & The Vandellas', 
                                         'Stills & Nash', 'OutKast','Bob Marley & The Wailers', 
                                         'Eric B. & Rakim', 'Martha Reeves & The Vandellas',
                                        'Drake', 'Rob Base & DJ E-Z Rock', 'UGK', 'Simon & Garfunkel',
                                         'Earth, Wind, & Fire', 'Grandmaster Flash & The Furious Five',
                                         'Smokey Robinson & The Miracles', 'The Notorious B.I.G.', 'Beyoncé', 
                                         'Bob Marley & The Wailers', 'Prince',"Guns N' Roses"])


lyric_df['Artist'] = lyric_df['Artist'].replace(['The Chicks','Neil Young & Crazy Horse', 'Muddy Waters And The Rolling Stones',
                                                'Run–DMC','KISS','Patti Smith Group','Jorge Ben Jor','\u200bThe Everly Brothers',
                                                'Funky 4 + 1', 'JAY-Z', 'Prince and the Revolution'],
                                                ['Dixie Chicks','Neil Young', 'Muddy Waters', 'Run-DMC','Kiss', 'Patti Smith',
                                                'Jorge Ben', 'Everly Brothers', 'The Funky 4 + 1', 'Jay-Z', 'Prince'])



#Rename values in Songs so RS500 and lyric_df match
RS500['Title'] = RS500['Title'].replace(['If I Aint Got You','Walk on By', 'Tutti-Frutti','Rappers Delight','Thats the Joint',
                                         'Stayin Alive','Youre So Vain','I Cant Help Myself (Sugar Pie,  Honey Bunch)',
                                         'Sunday Mornin Comin Down','Livin on a Prayer','In Da Club','We’re Goin Down',
                                         'La Vida Es un Carnaval','California Dreamin ','Say It Loud (I’m Black and I’m Proud)',
                                         'Im Coming Out','I Cant Make You Love Me','Merry Go Round','Dont Leave Me This Way',
                                         'Im a Believer','(White Man) in Hammersmith Palais','Papa Was a Rollin Stone',
                                         'Scenes From an Italian Restaurant','Aint No Sunshine','Wont Get Fooled Again',
                                         'Georgia on My Mind','Grindin','Ever Fallen in Love (With Someone You Shouldntve)',
                                         'Your Cheatin Heart','Free Fallin','Boys of Summer',
                                         'Signed, Sealed, Delivered Im Yours','I Cant Stand the Rain','Ill Take You There',
                                         'Baba ORiley','No Woman No Cry','Whats Love Got to Do With It',
                                         'Hold On,  Were Going Home','Freedom! 90','Blowin in the Wind',
                                         'Good Golly, Miss Molly','Intl Players Anthem (I Choose You)','Lets Stay Together',
                                         'Whatd I Say','Reach Out (Ill Be There)','I Walk the Line',
                                         '(Sittin On) the Dock of the Bay','(I Cant Get No) Satisfaction',
                                         " Roses, 'Welcome to the Jungle", " Roses, 'Sweet Child O' Mine"],
                                        ['If I Ain’t Got You','Walk On By','Tutti Frutti','Rapper’s Delight','That’s The Joint',
                                         'Stayin’ Alive','You’re So Vain','I Can’t Help Myself (Sugar Pie,  Honey Bunch)',
                                         'Sunday Mornin’ Comin’ Down','Livin’ on a Prayer','In da Club','We’re Goin’ Down',
                                         'La Vida Es Un Carnaval','California Dreamin’','Say It Loud: I’m Black and I’m Proud',
                                         'I’m Coming Out','I Can’t Make You Love Me','Merry Go ’Round','Don’t Leave Me This Way',
                                         'I’m a Believer','(White Man) In Hammersmith Palais','Papa Was a Rollin’ Stone',
                                         'Scenes from an Italian Restaurant','Ain’t No Sunshine','Won’t Get Fooled Again',
                                         'Georgia On My Mind','Grindin’','Ever Fallen in Love (With Someone You Shouldn’t’ve)',
                                         'Your Cheatin’ Heart','Free Fallin’','The Boys of Summer',"Signed, Sealed, Delivered I'm Yours",
                                         'I Can’t Stand the Rain','I’ll Take You There','Baba O’Riley','No Woman, No Cry',
                                         'What’s Love Got to Do with It','Hold On, We’re Going Home','Freedom! ’90',
                                         'Blowin’ in the Wind','Good Golly Miss Molly','Int’l Players Anthem (I Choose You)',
                                         'Let’s Stay Together','What’d I Say',"Reach Out, I'll Be There",'I Walk The Line',
                                         '(Sittin’ On) The Dock of the Bay','(I Can’t Get No) Satisfaction',
                                        'Welcome to The Jungle',"Sweet Child O' Mine"])



lyric_df['Title'] = lyric_df['Title'].replace(['Get Lucky (Daft Punk feat. Pharrell Williams Cover)','Savage Remix',
                                               'They Reminisce Over You (T.R.O.Y.)','Sucker M.C.’s (Live)',
                                               'I Wanna Dance with Somebody (Who Loves Me)','Single Ladies (Put a Ring on It)',
                                               '\u200bbad guy','Teen Age Riot','Will You Still Love Me Tomorrow','\u200bthank u, next',
                                               'Oh Bondage Up Yours!','Maybellene','Running Up That Hill (Kate Bush cover)',
                                               'Figaro / Papa’s Got a Brand New Bag'],
                                                ['Get Lucky','Savage (Remix)','They Reminisce Over You','Sucker MCs',
                                                 'I Wanna Dance With Somebody (Who Loves Me)',
                                                 'Single Ladies (Put a Ring On It)','Bad Guy','Teenage Riot',
                                                 'Will You Love Me Tomorrow','Thank U, Next','Oh Bondage! Up Yours!',
                                                 'Maybelline','Running Up That Hill','Papa’s Got a Brand New Bag'])



###########
df = pd.merge(RS500, lyric_df, on=["Title", "Artist"])

#df.head(500)


##############################
#######Count Vectorizing######
##############################


#Count Vectorizer
My_CV1=CountVectorizer(input='content',
                        stop_words='english',
                        lowercase=True,
                        min_df=2
                        #max_features=100                      
                        )

My_TF1=TfidfVectorizer(input='content',
                        stop_words='english',
                        lowercase=True,
                        min_df=2
                        #max_features=1000                      
                        )

My_Bern=CountVectorizer(input='content',
                        stop_words='english',
                        lowercase=True,
                        min_df=2,
                        binary=True
                        #max_features=1000                      
                        )

#Vectorize and use TFIDF
X_CV1=My_CV1.fit_transform(df['lyrics'])
X_TF1=My_TF1.fit_transform(df['lyrics'])
X_BNM = My_Bern.fit_transform(df['lyrics'])


ColNamesCV=My_CV1.get_feature_names_out()
ColNamesTF1=My_TF1.get_feature_names_out()
ColNamesBNM=My_Bern.get_feature_names_out()

## Create the dataframes
cv_df = pd.DataFrame(X_CV1.toarray(), columns=ColNamesCV)
tf_df = pd.DataFrame(X_TF1.toarray(), columns=ColNamesTF1)
bnm_df = pd.DataFrame(X_BNM.toarray(), columns=ColNamesBNM)

#print(cv_df)
#print(tf_df)
#print(bnm_df)

#Read out the files so we don't need to use the API every time
CV_File="/Users/joseph_davis/Desktop/Syracuse/Semester 4/Text Mining/Final/CV.csv"
TF_File="/Users/joseph_davis/Desktop/Syracuse/Semester 4/Text Mining/Final/Tfidf.csv"
BERN_FILE="/Users/joseph_davis/Desktop/Syracuse/Semester 4/Text Mining/Final/BERN.csv"

cv_df.to_csv(CV_File, index = False)
tf_df.to_csv(TF_File, index = False)
bnm_df.to_csv(BERN_FILE, index = False)


#The clean csv files are just the vectorized lyrics, will need to add labels
#For each classification problem
CleanCV = pd.read_csv("/Users/joseph_davis/Desktop/Syracuse/Semester 4/Text Mining/Final/CV.csv")
CleanTF = pd.read_csv("/Users/joseph_davis/Desktop/Syracuse/Semester 4/Text Mining/Final/Tfidf.csv")
CleanBERN = pd.read_csv("/Users/joseph_davis/Desktop/Syracuse/Semester 4/Text Mining/Final/BERN.csv")


#CleanCV
#CleanTF
#CleanBERN

#take a look at the columns we want to observe
for column in RS500:
    print(column)

## Add labels to dataframes
#100's label
cv100 = CleanCV
cv100['LABEL'] = RS500['100s_rank']
temp = cv100.columns.tolist()
new = temp[-1:] + temp[:-1]
cv100 = cv100[new]
#print(cv100)

#TFIDVectorizer
tf100 = CleanTF
tf100['LABEL'] = RS500['100s_rank']
temp = tf100.columns.tolist()
new = temp[-1:] + temp[:-1]
tf100=tf100[new]
#print(tf100)

#Countvectorizer binary
bn100 = CleanBERN
bn100['LABEL'] = RS500['100s_rank']
temp = bn100.columns.tolist()
new = temp[-1:] + temp[:-1]
bn100=bn100[new]
#print(bn100)




#50's label
#CV
cv50 = CleanCV
cv50['LABEL'] = RS500['50s_rank']
temp = cv50.columns.tolist()
new = temp[-1:] + temp[:-1]
cv50 = cv50[new]
#print(cv50)

#TFIDVectorizer
tf50 = CleanTF
tf50['LABEL'] = RS500['50s_rank']
temp = tf50.columns.tolist()
new = temp[-1:] + temp[:-1]
tf50=tf50[new]
#print(tf50)

#Countvectorizer binary
bn50 = CleanBERN
bn50['LABEL'] = RS500['50s_rank']
temp = bn50.columns.tolist()
new = temp[-1:] + temp[:-1]
bn50=bn50[new]
#print(bn50)




#Genre label
#CV
cvGenre = CleanCV
cvGenre['LABEL'] = RS500['Genre']
temp = cvGenre.columns.tolist()
new = temp[-1:] + temp[:-1]
cvGenre = cvGenre[new]
#print(cv100)

#TFIDVectorizer
tfGenre = CleanTF
tfGenre['LABEL'] = RS500['Genre']
temp = tfGenre.columns.tolist()
new = temp[-1:] + temp[:-1]
tfGenre=tfGenre[new]
#print(tf100)

#Countvectorizer binary
bnGenre = CleanBERN
bnGenre['LABEL'] = RS500['Genre']
temp = bnGenre.columns.tolist()
new = temp[-1:] + temp[:-1]
bnGenre=bnGenre[new]
#print(bn100)




#popularity label
#CV
cvPop = CleanCV
cvPop['LABEL'] = RS500['pop_grade']
temp = cvPop.columns.tolist()
new = temp[-1:] + temp[:-1]
cvPop = cvPop[new]
#print(cv100)

#TFIDVectorizer
tfPop = CleanTF
tfPop['LABEL'] = RS500['pop_grade']
temp = tfPop.columns.tolist()
new = temp[-1:] + temp[:-1]
tfPop=tfPop[new]
#print(tf100)

#Countvectorizer binary
bnPop = CleanBERN
bnPop['LABEL'] = RS500['pop_grade']
temp = bnPop.columns.tolist()
new = temp[-1:] + temp[:-1]
bnPop=bnPop[new]
#print(bn100)




#decade label
#CV
cvdec = CleanCV
cvdec['LABEL'] = RS500['decade']
temp = cvdec.columns.tolist()
new = temp[-1:] + temp[:-1]
cvdec = cvdec[new]
#print(cv100)

#TFIDVectorizer
tfdec = CleanTF
tfdec['LABEL'] = RS500['decade']
temp = tfdec.columns.tolist()
new = temp[-1:] + temp[:-1]
tfdec=tfdec[new]
#print(tf100)

#Countvectorizer binary
bndec = CleanBERN
bndec['LABEL'] = RS500['decade']
temp = bndec.columns.tolist()
new = temp[-1:] + temp[:-1]
bndec=bndec[new]
#print(bn100)

###################################
#########Machine Learning##########
###################################
#100's split (Honestly all really bad)
#Splitting into training and testing datasets
#CV100
train_cv100, test_cv100 = train_test_split(cv100, test_size=0.33, random_state=42)
trainLabel_cv100 = train_cv100['LABEL']
testLabel_cv100 = test_cv100['LABEL']
train_cv100 = train_cv100.drop(columns='LABEL')
test_cv100 = test_cv100.drop(columns='LABEL')

#TF100
train_tf100, test_tf100 = train_test_split(tf100, test_size=0.33, random_state=42)
trainLabel_tf100 = train_tf100['LABEL']
testLabel_tf100 = test_tf100['LABEL']
train_tf100 = train_tf100.drop(columns='LABEL')
test_tf100 = test_tf100.drop(columns='LABEL')

#BN100
train_bn100, test_bn100 = train_test_split(bn100, test_size=0.33, random_state=42)
trainLabel_bn100 = train_bn100['LABEL']
testLabel_bn100 = test_bn100['LABEL']
train_bn100 = train_bn100.drop(columns='LABEL')
test_bn100 = test_bn100.drop(columns='LABEL')

#Decision Trees
clf = tree.DecisionTreeClassifier(max_depth=5)
#Decision Tree cv100
clfcv100 = clf.fit(train_cv100, trainLabel_cv100)
Prediction1=clfcv100.predict(test_cv100)
matrix1 = confusion_matrix(testLabel_cv100, Prediction1)
matrix1
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfcv100, fontsize=12, feature_names=ColNamesCV)
plt.show()




#Decision Tree tf100
clftf100 = clf.fit(train_tf100, trainLabel_tf100)
Prediction2=clftf100.predict(test_tf100)
matrix2 = confusion_matrix(testLabel_tf100, Prediction2)
matrix2
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clftf100, fontsize=12, feature_names=ColNamesTF1)
plt.show()
len(test_tf100)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_names100=["Top 100", "100-200", "200-300", "300-400", "400-500"]

confusion1 = confusion_matrix(testLabel_tf100,Prediction2)
np.set_printoptions(precision=2)
# Plot confusion matrix
tf100=ConfusionMatrixDisplay(confusion1, display_labels=class_names100)
tf100.plot(cmap = plt.cm.Oranges)




#Decision Tree Bernoulli
clfbn100 = clf.fit(train_bn100, trainLabel_bn100)
Prediction3=clfbn100.predict(test_bn100)
matrix3 = confusion_matrix(testLabel_bn100, Prediction3)
matrix3
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfbn100, fontsize=12, feature_names=ColNamesBNM)
plt.show()

#MNB cv100
MyModelNB1 = MultinomialNB()
MyModelNB1.fit(train_cv100, trainLabel_cv100)
Prediction16 = MyModelNB1.predict(test_cv100)
matrix16 = confusion_matrix(testLabel_cv100, Prediction16)
matrix16

#MNBtf100
MyModelNB2 = MultinomialNB()
MyModelNB2.fit(train_tf100, trainLabel_tf100)
Prediction17 = MyModelNB2.predict(test_tf100)
matrix17 = confusion_matrix(testLabel_tf100, Prediction17)
matrix17

#MNBbn100
MyModelNB3 = MultinomialNB()
MyModelNB3.fit(train_bn100, trainLabel_bn100)
Prediction18 = MyModelNB3.predict(test_bn100)
matrix18 = confusion_matrix(testLabel_bn100, Prediction18)
matrix18


confusion6 = confusion_matrix(testLabel_tf100,Prediction18)
np.set_printoptions(precision=2)
# Plot confusion matrix
tf100=ConfusionMatrixDisplay(confusion6, display_labels=class_names100)
tf100.plot(cmap = plt.cm.Oranges)



print('Accuracy for Bernoulli Naive Bayes:', accuracy_score(testLabel_bn100, Prediction18))
print('F1 score:', f1_score(testLabel_bn100, Prediction18, average="macro"))
print(classification_report(testLabel_bn100, Prediction18))





#50's split
#Splitting into training and testing datasets
#CV50
train_cv50, test_cv50 = train_test_split(cv50, test_size=0.33, random_state=42)
trainLabel_cv50 = train_cv50['LABEL']
testLabel_cv50 = test_cv50['LABEL']
train_cv50 = train_cv50.drop(columns='LABEL')
test_cv50 = test_cv50.drop(columns='LABEL')

#TF50
train_tf50, test_tf50 = train_test_split(tf50, test_size=0.33, random_state=42)
trainLabel_tf50 = train_tf50['LABEL']
testLabel_tf50 = test_tf50['LABEL']
train_tf50 = train_tf50.drop(columns='LABEL')
test_tf50 = test_tf50.drop(columns='LABEL')

#BN50
train_bn50, test_bn50 = train_test_split(bn50, test_size=0.33, random_state=42)
trainLabel_bn50 = train_bn50['LABEL']
testLabel_bn50 = test_bn50['LABEL']
train_bn50 = train_bn50.drop(columns='LABEL')
test_bn50 = test_bn50.drop(columns='LABEL')

#Decision Trees
clf = tree.DecisionTreeClassifier(max_depth=5)
#Decision Tree cv50
clfcv50 = clf.fit(train_cv50, trainLabel_cv50)
Prediction4=clfcv50.predict(test_cv50)
matrix4 = confusion_matrix(testLabel_cv50, Prediction4)
matrix4
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfcv50, fontsize=12, feature_names=ColNamesCV)
plt.show()


class_names50=["50-100","100-150" ,"150-200", "200-250","250-300", 
                "300-350","350-400","400-450", "450-500"]
confusion2 = confusion_matrix(testLabel_cv50,Prediction4)
np.set_printoptions(precision=2)
# Plot confusion matrix
cv50=ConfusionMatrixDisplay(confusion2, display_labels=class_names50)
cv50.plot(cmap = plt.cm.Oranges)



#Decision Tree tf50
clftf50 = clf.fit(train_tf50, trainLabel_tf50)
Prediction5=clftf50.predict(test_tf50)
matrix5 = confusion_matrix(testLabel_tf50, Prediction5)
matrix5
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clftf50, fontsize=12, feature_names=ColNamesTF1)
plt.show()
len(test_tf50)

#Decision Tree Bernoulli
clfbn50 = clf.fit(train_bn50, trainLabel_bn50)
Prediction6=clfbn50.predict(test_bn50)
matrix6 = confusion_matrix(testLabel_bn50, Prediction6)
matrix6
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfbn50, fontsize=12, feature_names=ColNamesBNM)
plt.show()



#MNB cv50
MyModelNB4 = MultinomialNB()
MyModelNB4.fit(train_cv50, trainLabel_cv50)
Prediction19 = MyModelNB4.predict(test_cv50)
matrix19 = confusion_matrix(testLabel_cv50, Prediction19)
matrix19


confusion7 = confusion_matrix(testLabel_cv50,Prediction19)
np.set_printoptions(precision=2)
# Plot confusion matrix
cv50=ConfusionMatrixDisplay(confusion7, display_labels=class_names50)
cv50.plot(cmap = plt.cm.Oranges)
plt.xticks(rotation = 45, ha = 'right')
plt.show()



print('Accuracy for Count Vectorizer Naive Bayes:', accuracy_score(testLabel_cv50, Prediction19))
print('F1 score:', f1_score(testLabel_cv50, Prediction19, average="macro"))
print(classification_report(testLabel_cv50, Prediction19))



#MNBtf50
MyModelNB5 = MultinomialNB()
MyModelNB5.fit(train_tf50, trainLabel_tf50)
Prediction20 = MyModelNB5.predict(test_tf50)
matrix20 = confusion_matrix(testLabel_tf50, Prediction20)
matrix20

#MNBbn50
MyModelNB6 = MultinomialNB()
MyModelNB6.fit(train_bn50, trainLabel_bn50)
Prediction21 = MyModelNB6.predict(test_bn50)
matrix21 = confusion_matrix(testLabel_bn50, Prediction21)
matrix21



#GENRE split
#Splitting into training and testing datasets
#CVGenre
train_cvGenre, test_cvGenre = train_test_split(cvGenre, test_size=0.33, random_state=42)
trainLabel_cvGenre = train_cvGenre['LABEL']
testLabel_cvGenre = test_cvGenre['LABEL']
train_cvGenre = train_cvGenre.drop(columns='LABEL')
test_cvGenre = test_cvGenre.drop(columns='LABEL')

#TFGenre
train_tfGenre, test_tfGenre = train_test_split(tfGenre, test_size=0.33, random_state=42)
trainLabel_tfGenre = train_tfGenre['LABEL']
testLabel_tfGenre = test_tfGenre['LABEL']
train_tfGenre = train_tfGenre.drop(columns='LABEL')
test_tfGenre = test_tfGenre.drop(columns='LABEL')

#BNGenre
train_bnGenre, test_bnGenre = train_test_split(bnGenre, test_size=0.33, random_state=42)
trainLabel_bnGenre = train_bnGenre['LABEL']
testLabel_bnGenre = test_bnGenre['LABEL']
train_bnGenre = train_bnGenre.drop(columns='LABEL')
test_bnGenre = test_bnGenre.drop(columns='LABEL')

#Decision Trees
clf = tree.DecisionTreeClassifier(max_depth=100)
#Decision Tree cvGenre
clfcvGenre = clf.fit(train_cvGenre, trainLabel_cvGenre)
Prediction7=clfcvGenre.predict(test_cvGenre)
matrix7 = confusion_matrix(testLabel_cvGenre, Prediction7)
matrix7
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfcvGenre, fontsize=12, feature_names=ColNamesCV)
plt.show()

#Decision Tree tfGenre
clftfGenre = clf.fit(train_tfGenre, trainLabel_tfGenre)
Prediction8=clftfGenre.predict(test_tfGenre)
matrix8 = confusion_matrix(testLabel_tfGenre, Prediction8)
matrix8
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clftfGenre, fontsize=12, feature_names=ColNamesTF1)
plt.show()

#Decision Tree Bernoulli Genre
clfbnGenre = clf.fit(train_bnGenre, trainLabel_bnGenre)
Prediction9=clfbnGenre.predict(test_bnGenre)
matrix9 = confusion_matrix(testLabel_bnGenre, Prediction9)
matrix9
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfbnGenre, fontsize=12, feature_names=ColNamesBNM)
plt.show()
len(test_tfGenre)


class_namesGenre = ["Blues", "Country", "Disco", "Electronic", "Folk", "Gospel", "Hip-Hop", "Metal", "Pop", "R&B", "Reggae", "Rock"]
confusion3 = confusion_matrix(testLabel_bnGenre,Prediction9)
np.set_printoptions(precision=2, )
# Plot confusion matrix
bnGenre=ConfusionMatrixDisplay(confusion3, display_labels=class_namesGenre)
bnGenre.plot(cmap = plt.cm.Oranges)
plt.xticks(rotation = 45, ha = 'right')
plt.show()

print(classification_report(testLabel_bnGenre, Prediction9))


#MNBcvGenre
MyModelNB7 = MultinomialNB()
MyModelNB7.fit(train_cvGenre, trainLabel_cvGenre)
Prediction22 = MyModelNB7.predict(test_cvGenre)
matrix22 = confusion_matrix(testLabel_cvGenre, Prediction22)
matrix22

#MNBtfGenre
MyModelNB8 = MultinomialNB()
MyModelNB8.fit(train_tfGenre, trainLabel_tfGenre)
Prediction23 = MyModelNB8.predict(test_tfGenre)
matrix23 = confusion_matrix(testLabel_tfGenre, Prediction23)
matrix23

confusion8 = confusion_matrix(testLabel_tfGenre,Prediction23)
np.set_printoptions(precision=2)
# Plot confusion matrix
tfgenre=ConfusionMatrixDisplay(confusion8, display_labels=class_namesGenre)
tfgenre.plot(cmap = plt.cm.Oranges)
plt.xticks(rotation = 45, ha = 'right')
plt.show()


print('Accuracy for TFIDF Naive Bayes:', accuracy_score(testLabel_tfGenre, Prediction23))
print('F1 score:', f1_score(testLabel_tfGenre, Prediction23, average="macro"))
print(classification_report(testLabel_tfGenre, Prediction23))


#MNBbnGenre
MyModelNB9 = MultinomialNB()
MyModelNB9.fit(train_bnGenre, trainLabel_bnGenre)
Prediction24 = MyModelNB9.predict(test_bnGenre)
matrix24 = confusion_matrix(testLabel_bnGenre, Prediction24)
matrix24

model3results = sklearn.metrics.precision_recall_fscore_support(testLabel_tfGenre, Prediction23)
model3results


#popularity split
#Splitting into training and testing datasets
#CV
train_cvPop, test_cvPop = train_test_split(cvPop, test_size=0.33, random_state=42)
trainLabel_cvPop = train_cvPop['LABEL']
testLabel_cvPop = test_cvPop['LABEL']
train_cvPop = train_cvPop.drop(columns='LABEL')
test_cvPop = test_cvPop.drop(columns='LABEL')

#TF
train_tfPop, test_tfPop = train_test_split(tfPop, test_size=0.33, random_state=42)
trainLabel_tfPop = train_tfPop['LABEL']
testLabel_tfPop = test_tfPop['LABEL']
train_tfPop = train_tfPop.drop(columns='LABEL')
test_tfPop = test_tfPop.drop(columns='LABEL')

#BN
train_bnPop, test_bnPop = train_test_split(bnPop, test_size=0.33, random_state=42)
trainLabel_bnPop = train_bnPop['LABEL']
testLabel_bnPop = test_bnPop['LABEL']
train_bnPop = train_bnPop.drop(columns='LABEL')
test_bnPop = test_bnPop.drop(columns='LABEL')

#Decision Trees
clf = tree.DecisionTreeClassifier(max_depth=100)
#Decision Tree CV Popularity
clfcvPop = clf.fit(train_cvPop, trainLabel_cvPop)
Prediction10=clfcvPop.predict(test_cvPop)
matrix10 = confusion_matrix(testLabel_cvPop, Prediction10)
matrix10
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfcvPop, fontsize=12, feature_names=ColNamesCV)
plt.show()

#Decision Tree TFIDF Popularity
clftfPop = clf.fit(train_tfPop, trainLabel_tfPop)
Prediction11=clftfPop.predict(test_tfPop)
matrix11 = confusion_matrix(testLabel_tfPop, Prediction11)
matrix11
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clftfPop, fontsize=12, feature_names=ColNamesTF1)
plt.show()

#Decision Tree Bernoulli Popularity
clfbnPop = clf.fit(train_bnPop, trainLabel_bnPop)
Prediction12=clfbnPop.predict(test_bnPop)
matrix12 = confusion_matrix(testLabel_bnPop, Prediction12)
matrix12
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfbnGenre, fontsize=12, feature_names=ColNamesBNM)
plt.show()
len(test_tfPop)

class_namesPop = class_namespop=["B", "C", "D", "F"]
confusion4 = confusion_matrix(testLabel_bnPop,Prediction12)
np.set_printoptions(precision=2, )
# Plot confusion matrix
bnGenre=ConfusionMatrixDisplay(confusion4, display_labels=class_namesPop)
bnGenre.plot(cmap = plt.cm.Oranges)
plt.show()



#MNBcvPop
MyModelNB10 = MultinomialNB()
MyModelNB10.fit(train_cvPop, trainLabel_cvPop)
Prediction25 = MyModelNB10.predict(test_cvPop)
matrix25 = confusion_matrix(testLabel_cvPop, Prediction25)
matrix25


#MNBtfPop
MyModelNB11 = MultinomialNB()
MyModelNB11.fit(train_tfPop, trainLabel_tfPop)
Prediction26 = MyModelNB11.predict(test_tfPop)
matrix26 = confusion_matrix(testLabel_tfPop, Prediction26)
matrix26

confusion9 = confusion_matrix(testLabel_tfPop,Prediction26)
np.set_printoptions(precision=2)
# Plot confusion matrix
tfPop=ConfusionMatrixDisplay(confusion9, display_labels=class_namesPop)
tfPop.plot(cmap = plt.cm.Oranges)



print('Accuracy for TFIDF Naive Bayes:', accuracy_score(testLabel_tfPop, Prediction26))
print('F1 score:', f1_score(testLabel_tfPop, Prediction26, average="macro"))
print(classification_report(testLabel_tfPop, Prediction26))


#MNBbnPop
MyModelNB12 = MultinomialNB()
MyModelNB12.fit(train_bnPop, trainLabel_bnPop)
Prediction27 = MyModelNB12.predict(test_bnPop)
matrix27 = confusion_matrix(testLabel_bnPop, Prediction27)
matrix27


#decade split
#Splitting into training and testing datasets
#CV
train_cvdec, test_cvdec = train_test_split(cvdec, test_size=0.33, random_state=42)
trainLabel_cvdec = train_cvdec['LABEL']
testLabel_cvdec = test_cvdec['LABEL']
train_cvdec = train_cvdec.drop(columns='LABEL')
test_cvdec = test_cvdec.drop(columns='LABEL')

#TF
train_tfdec, test_tfdec = train_test_split(tfdec, test_size=0.33, random_state=42)
trainLabel_tfdec = train_tfdec['LABEL']
testLabel_tfdec = test_tfdec['LABEL']
train_tfdec = train_tfdec.drop(columns='LABEL')
test_tfdec = test_tfdec.drop(columns='LABEL')

#BN
train_bndec, test_bndec = train_test_split(bndec, test_size=0.33, random_state=42)
trainLabel_bndec = train_bndec['LABEL']
testLabel_bndec = test_bndec['LABEL']
train_bndec = train_bndec.drop(columns='LABEL')
test_bndec = test_bndec.drop(columns='LABEL')

#Decision Trees
clf = tree.DecisionTreeClassifier(max_depth=100)
#Decision Tree CV Decade
clfcvdec = clf.fit(train_cvdec, trainLabel_cvdec)
Prediction13=clfcvdec.predict(test_cvdec)
matrix13 = confusion_matrix(testLabel_cvdec, Prediction13)
matrix13
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfcvdec, fontsize=12, feature_names=ColNamesCV)
plt.show()


class_namesdec=["00s", "10s", "30s", "40s", "50s", "60s", "70s", "80s", "90s"]
confusion5 = confusion_matrix(testLabel_cvdec,Prediction13)
np.set_printoptions(precision=2, )
# Plot confusion matrix
cvdec=ConfusionMatrixDisplay(confusion5, display_labels=class_namesdec)
cvdec.plot(cmap = plt.cm.Oranges)
plt.show()

#Decision Tree TFIDF Decade
clftfdec = clf.fit(train_tfdec, trainLabel_tfdec)
Prediction14=clftfdec.predict(test_tfdec)
matrix14 = confusion_matrix(testLabel_tfdec, Prediction14)
matrix14
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clftfdec, fontsize=12, feature_names=ColNamesTF1)
plt.show()

#Decision Tree Bernoullio Decade
clfbndec = clf.fit(train_bndec, trainLabel_bndec)
Prediction15=clfbndec.predict(test_bndec)
matrix15 = confusion_matrix(testLabel_bndec, Prediction15)
matrix15
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(clfbndec, fontsize=12, feature_names=ColNamesBNM)
plt.show()
len(test_tfdec)


#MNBcvdec
MyModelNB13 = MultinomialNB()
MyModelNB13.fit(train_cvdec, trainLabel_cvdec)
Prediction28 = MyModelNB13.predict(test_cvdec)
matrix28 = confusion_matrix(testLabel_cvdec, Prediction28)
matrix28


#MNBtfdec
MyModelNB14 = MultinomialNB()
MyModelNB14.fit(train_tfdec, trainLabel_tfdec)
Prediction29 = MyModelNB14.predict(test_tfdec)
matrix29 = confusion_matrix(testLabel_tfdec, Prediction29)
matrix29


#MNBbndec
MyModelNB15 = MultinomialNB()
MyModelNB15.fit(train_bndec, trainLabel_bndec)
Prediction30 = MyModelNB15.predict(test_bndec)
matrix30 = confusion_matrix(testLabel_bndec, Prediction30)
matrix30

for line in testLabel_bndec:
    print(line)
class_namesdec=["00s", "10s", "30s", "50s", "60s", "70s", "80s", "90s"]
confusion10 = confusion_matrix(testLabel_bndec,Prediction30)
np.set_printoptions(precision=2)
# Plot confusion matrix
tfPop=ConfusionMatrixDisplay(confusion10, display_labels=class_namesdec)
tfPop.plot(cmap = plt.cm.Oranges)



print('Accuracy for Bernoulli/Binary Naive Bayes:', accuracy_score(testLabel_tfdec, Prediction30))
print('F1 score:', f1_score(testLabel_tfdec, Prediction30, average="macro"))
print(classification_report(testLabel_tfdec, Prediction30))



## Supervised Vector Machines (SVM)
#Create the first model
svm_model = LinearSVC(C=50)

 # Using 100s_rank
#Fit the model and print results
svm_100 = svm_model.fit(train_cv100, trainLabel_cv100)
p_svm_100 = svm_100.predict(test_cv100)

print("The SVM model predicted:")
print(p_svm_100)
print("The actual labels are:")
print(testLabel_cv100)
print('Accuracy for SVM:', accuracy_score(testLabel_cv100, p_svm_100))
print('F1 score for SVM:', f1_score(testLabel_cv100, p_svm_100, average="macro"))
print(classification_report(testLabel_cv100, p_svm_100))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#Confusion Matrix 100
class_names100=["Top 100", "100-200", "200-300", "300-400", "400-500"]

cnf_cv100 = confusion_matrix(testLabel_cv100,p_svm_100)
np.set_printoptions(precision=2)
# Plot confusion matrix
svm100=ConfusionMatrixDisplay(cnf_cv100, display_labels=class_names100)
svm100.plot(cmap = plt.cm.Oranges)
plt.title('Confusion Matrix For SVM with 100s Rank')
plt.show()



 # Using 50s_rank
#Fit the model and print results
svm_50 = svm_model.fit(train_cv50, trainLabel_cv50)
p_svm_50 = svm_50.predict(test_cv50)

print("The SVM model predicted:")
print(p_svm_50)
print("The actual labels are:")
print(testLabel_cv50)
print('Accuracy for SVM:', accuracy_score(testLabel_cv50, p_svm_50))
print('F1 score for SVM:', f1_score(testLabel_cv50, p_svm_50, average="macro"))
print(classification_report(testLabel_cv50, p_svm_50))

#Confusion Matrix 50
class_names50=["50-100","100-150" ,"150-200", "200-250","250-300", 
                "300-350","350-400","400-450", "450-500"]

cnf_cv50 = confusion_matrix(testLabel_cv50,p_svm_50)
np.set_printoptions(precision=2)
# Plot confusion matrix
svm50=ConfusionMatrixDisplay(cnf_cv50, display_labels=class_names50)
svm50.plot(cmap = plt.cm.Oranges)
plt.title('Confusion Matrix For SVM with 50s Rank')
plt.xticks(rotation = 45, ha = 'right')
plt.show()


 # Using Genre
#Fit the model and print results
svm_gen = svm_model.fit(train_cvGenre, trainLabel_cvGenre)
p_svm_gen = svm_gen.predict(test_cvGenre)

print("The SVM model predicted:")
print(p_svm_gen)
print("The actual labels are:")
print(testLabel_cvGenre)
print('Accuracy for SVM:', accuracy_score(testLabel_cvGenre, p_svm_gen))
print('F1 score for SVM:', f1_score(testLabel_cvGenre, p_svm_gen, average="macro"))
print(classification_report(testLabel_cvGenre, p_svm_gen))


 # Using Popularity
#Fit the model and print results
svm_pop = svm_model.fit(train_cvPop, trainLabel_cvPop)
p_svm_pop = svm_pop.predict(test_cvPop)

print("The SVM model predicted:")
print(p_svm_pop)
print("The actual labels are:")
print(testLabel_cvPop)
print('Accuracy for SVM:', accuracy_score(testLabel_cvPop, p_svm_pop))
print('F1 score for SVM:', f1_score(testLabel_cvPop, p_svm_pop, average="macro"))
print(classification_report(testLabel_cvPop, p_svm_pop))


#Confusion Matrix popularity
class_namespop=["B", "C", "D", "F"]

cnf_cvpop = confusion_matrix(testLabel_cvPop,p_svm_pop)
np.set_printoptions(precision=2)
# Plot confusion matrix
svmpop=ConfusionMatrixDisplay(cnf_cvpop, display_labels=class_namespop)
svmpop.plot(cmap = plt.cm.Oranges)
plt.title('Confusion Matrix For SVM with Popularity')
plt.show()

 # Using Decade
#Fit the model and print results
svm_dec = svm_model.fit(train_cvdec, trainLabel_cvdec)
p_svm_dec = svm_dec.predict(test_cvdec)

print("The SVM model predicted:")
print(p_svm_dec)
print("The actual labels are:")
print(testLabel_cvdec)
print('Accuracy for SVM:', accuracy_score(testLabel_cvdec, p_svm_dec))
print('F1 score for SVM:', f1_score(testLabel_cvdec, p_svm_dec, average="macro"))
print(classification_report(testLabel_cvdec, p_svm_dec))

#Confusion Matrix Decade
class_namesdec=["00s", "10s", "30s", "40s", "50s", "60s", "70s", "80s", "90s"]

cnf_cvdec = confusion_matrix(testLabel_cvdec,p_svm_dec)
np.set_printoptions(precision=2)
# Plot confusion matrix
svmdec=ConfusionMatrixDisplay(cnf_cvdec, display_labels=class_namesdec)
svmdec.plot(cmap = plt.cm.Oranges)
plt.title('Confusion Matrix For SVM with Decades')
plt.xticks(rotation = 45, ha = 'right')
plt.show()


from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

#Using LDA to Predict Genre
num_topics = 15

features = cvGenre.drop(columns = 'LABEL')
genre = cvGenre['LABEL']

LDA = LatentDirichletAllocation(n_components=num_topics)
LDA_CV = LDA.fit_transform(features)

print("SIZE: ", LDA_CV.shape)  

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Doc in data: ")
print(LDA_CV[0])
print("Seventh Doc in data: ")
print(LDA_CV[6])
print("List of prob: ")
print(LDA_CV)

# Visualize 15 topics
word_topic = np.array(LDA.components_)
word_topic = word_topic.transpose()

num_top_words = 15
vocab_array = np.asarray(ColNamesCV)

fontsize_base = 5

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)
    plt.ylim(0, num_top_words + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)

plt.tight_layout()
plt.show()













