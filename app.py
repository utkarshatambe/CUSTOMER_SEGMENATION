import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# page setup
st.set_page_config(page_icon="👥",page_title="CUSTOMER SEGMENTAION",layout="wide")

# load the data
st.subheader('UPLOAD DATA')
st.write('(required column - Annual Income(k$),Spending Score (1-100)')
file = st.file_uploader(" ",type =["csv"])
df=None
if file:
    df =pd.read_csv(file)


with st.sidebar:
    st.title("CUSTOMER SEGMENATION")
    st.image('img1.jpg')

    if df is not None:
        features = st.multiselect("select features: ",options=df.columns,default=["Annual Income (k$)","Spending Score (1-100)"])
        df =df.loc[:,features]
        
def preprocessing(df):
    # encoding
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype==object:
            df[col] = encoder.fit_transform(df[col])

def elbow():
    out=[]
    k_values = range(1,11)


    for i in k_values:
            model = KMeans(n_clusters=i)
            model.fit(df)
            out.append(model.inertia_)



    KL = KneeLocator(k_values ,out,curve='convex',direction = "decreasing")
    df1 =pd.DataFrame({"k_val":k_values,"inertia":out})


    st.subheader("ELBOW CURVE")
    fig = st.line_chart(data = df1,x = "k_val",y ="inertia")

    return KL.elbow

if df is not None:
     st.subheader("samples of the data uploaded for visulaization and clustering")
     st.write(df.sample(10))

     preprocessing(df)


    #  optimized k value
     k=elbow()

    # model training
     model = KMeans(n_clusters=k)
     model.fit(df)
     lables = model.labels_
     df['clusters'] = lables


    #  visualization
     st.subheader("CLUSTERD DATA")
     st.scatter_chart(data = df,x="Annual Income (k$)", y = "Spending Score (1-100)",color = "clusters")


  







