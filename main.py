import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(layout="wide")

# functions
def get_ratings(df):    
        model = "nlptown/bert-base-multilingual-uncased-sentiment"
                
        model_pipeline = pipeline("sentiment-analysis", model = model, tokenizer = model, truncation=True)
                
        def rate_comment(text):
            output = model_pipeline(text)[0]
            return output['label']
        
        rate_map = {'1 star':1,'2 stars':2,'3 stars':3,'4 stars':4,'5 stars': 5}
        
        # Column rating hasn't been created
        if 'rating' not in df.columns:
                
            df['rating'] = df['user_review'].map(rate_comment)    
            df['rating'] = df['rating'].map(rate_map)
        else:
            mask = df['rating'].isnull()
            df.loc[mask, 'rating'] = df.loc[mask, 'user_review'].map(rate_comment)
            df.loc[mask, 'rating'] = df.loc[mask, 'rating'].map(rate_map)
            
        return df    
    
def add_rows(df,user_name,product_name,user_review):
    # create the new row
    new_row = {'user_name': user_name, 'product_name': product_name,'user_review': user_review}
    # convert to dataframe
    new_row = pd.DataFrame([new_row])
    # Append to the DataFrame
    new_df = pd.concat([df,new_row],ignore_index=True)
    return new_df  

def plot_chart(df):
    
    product_df = df.groupby('product_name')['rating'].mean().reset_index().sort_values(by='rating',ascending=False)
    # Create the Altair chart
    font_props_x = alt.Axis(labelFont="Arial",titleFont="Arial",labelFontSize=12,titleFontSize=15,tickSize=0,labelAngle=0)
    font_props_y = alt.Axis(labelFont="Arial",titleFont="Arial",labelFontSize=12,titleFontSize=15,tickSize=0,labelAngle=0)
    title = alt.TitleParams(text='Mean Rating by Product', anchor='middle')
    
    chart = alt.Chart(product_df).mark_bar(width=50).encode(
            x = alt.X('rating',title='Rating',scale=alt.Scale(domain=[0, 5]),axis=font_props_y),
            y = alt.Y('product_name',sort='-x',title='Product Name',axis=font_props_x),
            color =alt.Color('product_name',legend=None),
            tooltip=['product_name','rating']  
        ).properties(
            title = title,
            # height = 500, 
        ).configure_axis(
            grid=False               
        ).configure_title(
            font = 'Arial',
            fontSize = 17
        )    
    return chart

def create_table(df):
    mean_ratings = df.groupby('product_name')['rating'].mean().reset_index(name='mean_rating')
    rating_count = df.groupby(['product_name','rating']).size().unstack(fill_value=0).reset_index()
    product_ratings_df = rating_count.merge(mean_ratings,on = 'product_name').sort_values(by = ['mean_rating'],ascending=False)
    
    product_ratings_df = product_ratings_df.rename(columns = {1:'1 star',2:'2 stars',3:'3 stars',4:'4 stars',5:'5 stars'})
    product_ratings_df = product_ratings_df.set_index('product_name')
    return product_ratings_df.loc[:,['5 stars','4 stars','3 stars','2 stars','1 star']]
 
      

  

# Title
st.title("Product Review App")

df_file = st.file_uploader('Upload csv file containing the user reviews',type=['.csv'])

if df_file is not None:
    
    df = pd.read_csv(df_file)
    
    if 'df' not in st.session_state:
        st.session_state['df'] = df 

    df_table = st.dataframe(st.session_state.df,use_container_width=True)


    #Sidebar to submit information
    formReview = st.sidebar.form(key="review",clear_on_submit=True)


    formReview.header("Enter your review")

    user_name = formReview.text_input("Enter your username:")
    product_name = formReview.selectbox("Select product:",options=df['product_name'].unique().tolist())
    user_review = formReview.text_area("Enter your review:")

    submit_button = formReview.form_submit_button("Submit")


    if submit_button:
        st.session_state.df = add_rows(st.session_state.df, user_name,product_name,user_review)
        df_table.dataframe(st.session_state.df,use_container_width=True) 

    col1,col2,col3 = st.columns([1,1,1])

    analyze_reviews = col2.button("Analyze Reviews")    
    

    if analyze_reviews:
        
        with st.spinner('Analyzing Reviews ⌛'):               
            st.session_state.df = get_ratings(st.session_state.df)
        
        df_table.dataframe(st.session_state.df,use_container_width=True)
        
        container = st.container(border=True)
        
        col1,col2,col3 = st.columns([2,5,4])
        
        tile1 = col1.container(border=True)
        tile1.metric(label="5 star ratings :heart_eyes:",value=f"{round(st.session_state.df['rating'].value_counts()[5]/len(st.session_state.df)*100,1)}%")
        tile1.metric(label="4 star ratings :smiley:",value=f"{round(st.session_state.df['rating'].value_counts()[4]/len(st.session_state.df)*100,1)}%")
        tile1.metric(label="3 star ratings :neutral_face:",value=f"{round(st.session_state.df['rating'].value_counts()[3]/len(st.session_state.df)*100,1)}%")
        tile1.metric(label="2 star ratings :angry:",value=f"{round(st.session_state.df['rating'].value_counts()[2]/len(st.session_state.df)*100,1)}%")
        tile1.metric(label="1 star ratings :rage:",value=f"{round(st.session_state.df['rating'].value_counts()[1]/len(st.session_state.df)*100,1)}%")
        
        tile2 = col2.container(border=True)
        chart = plot_chart(st.session_state.df)   
        tile2.altair_chart(chart,use_container_width=True)
        
        tile3 = col3.container(border=False)   
        product_df = create_table(st.session_state.df)
        tile3.dataframe(product_df)
                    
          
        

