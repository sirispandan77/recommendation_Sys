from prometheus_client import Histogram
import streamlit as st
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from PIL import Image
import plotly.figure_factory as ff
from PIL import Image
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel
from sentiment_analysis import get_sentimentAnalysis


#python code

#assign paths and directories:
basepath="C:/Users/Hp/OneDrive/Documents/majorproject/resources/"
filename1="positive_words.txt"
filename2="negative_words.txt"
filename3="all_words.txt"
filename4="data.csv"
filename5="visual_data.csv"
filename6="totaldata.csv"
filename7="emotionDetection_model.sav"

#***********************************************************************************
#content files
filename9 = 'tfidf_matrix.txt'
filename10 = 'indices.txt'
filename11 = 'train_data.txt'



#load models/data
positive=pickle.load(open(basepath+filename1,'rb'))
negative=pickle.load(open(basepath+filename2,'rb'))
allwrds=pickle.load(open(basepath+filename3,'rb'))
data=pickle.load(open(basepath+filename4,'rb'))
visual_Data=pickle.load(open(basepath+filename5,'rb'))
totaldata=pickle.load(open(basepath+filename6,'rb'))
emotion=pickle.load(open(basepath+filename7,'rb'))
#**********************************************************************************
#content-based
tmatrix = pickle.load(open(basepath+filename9, 'rb'))
indx= pickle.load(open(basepath+filename10, 'rb'))
train =pickle.load(open(basepath+filename11, 'rb'))





def all_wordcloud(words,color):
    wordcloud = WordCloud(width = 4000, height = 3000, background_color = color, max_words = 250).generate(words)
    fig,axes=plt.subplots(1,1)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, left=False,labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False,
                left=False, labelleft=False)
    axes.imshow(wordcloud, interpolation='bilinear')
    st.pyplot(fig)
    

def dataOverview():
    dat=pickle.load(open(basepath+filename6,'rb'))
    return dat.head()

def countPlot():
    fig = plt.figure(figsize=(5, 5))
    labels=[1,2,3,4,5]
    print(data["Rating"].value_counts())
    sns.countplot(x = labels, data = data["Rating"].value_counts())
    #sns.countplot(x = 'Rating', data = data)
    plt.ylim(0, 2)
    #plt.show()
    st.pyplot(fig)

def HeatMap():
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(visual_Data.corr(), annot=True);
    st.pyplot(fig)

def histogram():
    fig = plt.figure(figsize=(10, 4))
    #px.histogram(visual_Data, x = visual_Data.total_votes,y= visual_Data.helpful_votes ,marginal="rug" ,nbins=25,text_auto=True)
    plt.barh(visual_Data.total_votes,visual_Data.helpful_votes)
    #plt.hist(visual_Data.helpful_votes)
    plt.xlabel("Total votes")
    plt.ylabel("Helpful Votes")    
    plt.title("Total votes vs helpful votes")    
    st.balloons()
    st.pyplot(fig)

def Bargraph():
    fig = plt.figure(figsize = (10, 5))
    plt.bar(visual_Data.vine,visual_Data.verified_purchase)
    plt.xlabel("vine")
    plt.ylabel("verified_purchase")
    #st.snow()
    st.pyplot(fig)

def pieChart():    
    fig = plt.figure(figsize=(10, 3))
    labels=[1,2,3,4,5]
    plt.pie(x=data['Rating'].value_counts(), labels = labels,autopct='%.1f%%')
   # plt.ylabel("verified_purchase")
    plt.tight_layout()
    st.balloons()
    st.pyplot(fig)
#========================================================================================================================================================================================

def Emotiondetection(detect_text):
        pred,prob=get_sentimentAnalysis(detect_text)
        fig, ax = plt.subplots(figsize=(5, 3))
        if pred[0] == 1:
            text = 'Positive'
            class_proba = 100 * round(prob[0][1], 2)
            color = 'seagreen'
        else:
            text = 'Negative'
            class_proba = 100 * round(prob[0][0], 2)
            color = 'crimson'
        ax.text(0.5, 0.5, text, fontsize=50, ha='center', color=color)
        ax.text(0.5, 0.20, str(class_proba) + '%', fontsize=14, ha='center')
        ax.axis('off')
        ax.set_title('Sentiment Analysis', fontsize=14)
        st.pyplot(fig)




#============================================================================================================================================================================


#content-based
tfidf = TfidfVectorizer(stop_words='english')

# Compute the cosine similarity matrix
cos = linear_kernel(tmatrix, tmatrix)
def get_recommendations(name, cosine_sim=cos):
    # Get the index of the prod that matches the name
    idx = indx[name]
    # Get the pairwsie similarity scores of all products with that name
    print(idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar products
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    prod_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar products
    return train['Product Name'].iloc[prod_indices]




#=====================================================================================================================================================================================

header=  st.container()
dataset=  st.container()
options, visuals = st.columns([1,3])
backgroundColor="red"
primaryColor="purple"

def sideBar():
    graphs=["None","Bargraph", "heatmap" , "Piechart" , "Histogram", "countplot" ]
    side=st.sidebar.selectbox("Graphs",graphs)
    if( side=='None'):
        st.write()
        st.snow()
    elif side=='heatmap':
        st.subheader("heatmap")
        HeatMap()
    elif side=='Piechart':
        st.subheader('Piechart') 
        pieChart()    
    elif side=='Histogram':
        st.subheader('Histogram')
        histogram()   
    elif side=="Bargraph":
        st.subheader("Bargraph")
        Bargraph()     
    else:
        st.subheader('countplot')
        countPlot()

    #emotion detection
    detect_text=st.sidebar.text_input(label="enter Review to detect it's emotion")
    if(len(detect_text)>0):
        st.write("for the comment :")
        st.write(detect_text)
        Emotiondetection(detect_text)

    #content-based recommendation
    recommend=st.sidebar.text_input(label="enter category for product recommendation")
    if(len(recommend)>0):
        st.write("for the product: ")
        st.write(recommend)
        st.write(get_recommendations(recommend))
sideBar()


with header:
    st.title("Product recommendation ")
    with st.expander("Wondering what it is?"):
     st.write("""
         A product recommendation is basically a filtering system that seeks to predict and show the items that a user would like to purchase.If it shows you what you like then it is doing its job right.
     """)
     img1 = Image.open("C:/Users/Hp/OneDrive/Documents/majorproject/resources/recommendation2.png")
     img2= Image.open("C:/Users/Hp/OneDrive/Documents/majorproject/resources/recommendation3.png")
     st.image(img1,caption="E-commerce recommendation")
     st.image(img2,caption="Youtube recommendation")
    


with dataset:
    st.subheader("Overview of Dataset")
    with st.expander("Dive in"):
     st.write(dataOverview())
     st.write("""
                    -- Content

            - marketplace: 2 letter country code of the marketplace where the review was written.
            - customer_id: Random identifier that can be used to aggregate reviews written by a single author.
            - review_id: The unique ID of the review.
            - productid : The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same productid.
            - product_parent : Random identifier that can be used to aggregate reviews for the same product.
            - product_title:Title of the product.
            - product_category: Broad product category that can be used to group reviews(also used to group the dataset into coherent parts).
            - star_rating: The 1-5 star rating of the review.
            - helpful_votes: Number of helpful votes.
            - total_votes: Number of total votes the review received.
            - vine:  Review was written as part of the Vine program.
            - verified_purchase: The review is on a verified purchase.
            - review_headline: The title of the review.
            - review_body: The review text.
            - review_date: The date the review was written.
     """)


with options:
    st.subheader("Word Cloud")
    status = st.radio("select set of words ", ('None','positive', 'negative','all words'))

with visuals:
    st.subheader("")
    if (status == 'None'):
        st.write()
    elif status== 'negative':
        #st.write("neg")
        neg=all_wordcloud(negative,"black")
    elif status=='all words':
        all_wrds=all_wordcloud(allwrds,"white")
    else:
       #st.write("pos")
        pos=all_wordcloud(positive,"pink")
    
    








