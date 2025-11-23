
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import pickle
import streamlit as st
import os
import io

st.set_page_config(
    page_title="Cycling traffic in Paris",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Loading the random forest model
@st.cache_resource
def load_rf_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'rf_model.pkl')
    try:
        with open(model_path, 'rb') as best_rfmodel:
            rf_model = pickle.load(best_rfmodel)
        return rf_model
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        return None

rf_model = load_rf_model()

# Loading the original dataframe in the cache
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'df_original.csv')
    df = pd.read_csv(filepath, sep=';')
    return df

# Loading the preprocessed dataset in the cache
@st.cache_data
def preprocess_data(df):
    df_en = df.rename(columns={'Identifiant du compteur': 'Counter_ID',
                           'Nom du compteur': 'Counter_name',
                           'Identifiant du site de comptage' : 'Counter_site_ID',
                           'Nom du site de comptage' : 'Counter_site_name',
                           'Comptage horaire':'Hourly_counting',
                           'Date et heure de comptage' : 'Counting_date_hour',
                           'Date d\'installation du site de comptage' : 'Installation_date_of_the_counting_site',
                            'Lien vers photo du site de comptage' : 'Picture_URL',
                           'Coordonnées géographiques':"Geographic_coordinates",
                            'Identifiant technique compteur':"Counter_technical_ID",
                           'ID Photos' : "Picture_ID",
                            'test_lien_vers_photos_du_site_de_comptage_': "Test_URL_to_counting_site_picture",
                           'id_photo_1':'Picture_1_ID',
                           'url_sites':"Counter_picture_URL",
                           "type_dimage": "Image_format",
                           'mois_annee_comptage': "Month_year_counting_date"})
    df_en[['Latitude', 'Longitude']] = df_en['Geographic_coordinates'].str.split(',', expand=True)
    df_en['Latitude'] = pd.to_numeric(df_en['Latitude'])
    df_en['Longitude'] = pd.to_numeric(df_en['Longitude'])
    df_en['Counting_date_hour'] = pd.to_datetime(df_en['Counting_date_hour'], utc = True)
    df_en["counting_year"] = df_en["Counting_date_hour"].dt.year
    df_en["counting_month"] = df_en["Counting_date_hour"].dt.month
    df_en["counting_day"] = df_en["Counting_date_hour"].dt.day
    df_en["counting_hour"] = df_en["Counting_date_hour"].dt.hour
    df_en["counting_day_name"] = df_en["Counting_date_hour"].dt.day_name()
    df_en['weekday'] = df_en['counting_day_name'].apply(lambda x: 1 if x in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 0)
    df_en['weekend'] = df_en['counting_day_name'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    return df_en

# Define the app style
st.markdown("""
    <style>
        .stApp {
            background-color: #e0f7fa !important;
        }
        .stApp > header {
            background-color: transparent !important;
        }
        /* Forcing the font size */
        .stApp, .stApp div, .stApp span, .stApp p, .stApp ul, .stApp li, .stApp a {
            font-family: 'Roboto', Arial, sans-serif !important;
            font-size: 20px !important;
            color: #333333 !important;
        }
        .title {
            font-size: 40px !important;
            color: #006064 !important;
        }
        .header {
            font-size: 30px !important;
            color: #006064 !important;
        }
        .text {
            font-size: 20px !important;
            color: #333333 !important;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #006064 !important;
        }
        
        /* Styles for sidebar */
        [data-testid="stSidebar"] {
            background-color: #027cbd !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #027cbd !important;
        }
        [data-testid="stSidebar"] .sidebar-content {
            background-color: #027cbd !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {
            color: white !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a:hover {
            color: #e0f7fa !important;
            text-decoration: underline;
        }
        /* Style for the button */
        div.stButton > button:first-child {
            background-color: #e0f7fa !important;
            color: #026298 !important;
        }
        div.stButton > button:first-child:hover {
            background-color: #026298 !important;
            color: #026298 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Define the headers style
def underline_header(text, level):
    html = f"<h{level}><u>{text}</u></h{level}>"
    return st.markdown(html, unsafe_allow_html=True)

# Set title of the app
st.title("Cycling traffic in Paris")

# Sidebar title
st.sidebar.title("Summary")

# List of pages
pages = ["📄 Project description", "🔍 Data exploration", "📊 Data visualization", "🛠️ Data preparation", "🤖 Modelling", "🔮 Predictions"]

# Radio button to navigate pages
page = st.sidebar.radio("",pages)

# Adding authors' names in the sidebar
st.sidebar.markdown("### Authors")
st.sidebar.markdown("""
- Matea Mutz  <a href="https://www.linkedin.com/in/matea-mutz/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" height="20"></a>
- Sohith Varma Bhupathiraju  <a href="https://www.linkedin.com/in/sohith-varma-bhupathiraju/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" height="20"></a>
- Loris Durbano  <a href="https://www.linkedin.com/in/lorisdurbano/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" height="20"></a>
""", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
**Data source:** [Open Data Paris City](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/download/?format=csv&timezone=Europe/Paris&lang=fr&use_labels_for_header=true&csv_separator=)
""", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""**Program:** Data analysis bootcamp May, 2024""")

# Page content
if page == pages[0]:
    st.write("For many years, the City of Paris has been monitoring the growth of cycling with permanent bicycle counters.")
    st.text(" ")
    st.write("The project's first steps will be to analyze the information gathered by these bike counters (for 18 months between March, 2022 and May, 2024) and produce a visual representation of the timetables and affluent regions.")
    st.text(" ")
    st.write("The goal of this is to give Paris's town hall the means to assess what needs to be done to improve the city's numerous bike-friendly zones.")
    # Display the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'bike.jpg')
    if os.path.exists(image_path):
        st.image(image_path, caption='Cycling in Paris')
    else:
        st.error(f"Image {image_path} not found.")
    st.text(" ")
    st.write("""
    Based on the open datasets from Paris city, we will proceed to data exploration first then data visualization followed by the cleaning/processing steps.
    \nFinally, we will train several machine learning models and keep the most performant in order to deliver useful insights and recommendations to authorities regarding the high-traffic bicycle areas and periods.
    """)

if page == pages[1]:

    underline_header('Data exploration',2)

    # Data information
    df = load_data()
    with st.expander("Overview of the dataset"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    st.write("""
            We can observe a slightly different number of non-null values in the various columns, suggesting we will face some missing values.
            \n The dataset seems large enough to obtain insightful information since it has almost 1 million entries.""")

    # Data description
    underline_header("Descriptive statistics:", 2)
    with st.expander("Statistics"):
        st.write(df.describe())
    st.write("""We will not address the counting site id since it is not a relevant numerical variable.
             \nHowever, that’s interesting to observe our target variable distribution. 
             \n**High Variability**: The standard deviation is relatively high compared to the mean, indicating substantial variability in the hourly counts. There are many hours with counts far from the average.
             \n**Skewed Distribution**: The median (41) is much lower than the mean (75.8886), which suggests that the distribution of hourly counts is skewed. There are more hours with low counts, but a few hours with very high counts pull the mean up.
             \n**Presence of Outliers**: The minimum value is 0, and the maximum value is 8,190. The high maximum value indicates the presence of outliers or extreme values in the dataset.""")

    # Missing values percentage per column
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({'Variables': df.columns,'%': missing_percentage.round(2).apply(lambda x: f"{x:.2f}%")}).reset_index(drop=True)
    with st.expander("Missing Values Percentage"):
        st.table(missing_data)
    st.write("""The missing values in the dataset are relatively low and should be manageable without significant impact on your analysis.
             \nSome variables are likely to be without any impact on the target variable (such as the pictures the URL links for instance). 
             \nWe will address the missing geographical coordinates later depending on the missing values importance.""")

    # Renaming columns
    df_en = preprocess_data(df)
    underline_header("Translate the columns",2)
    st.write("The original dataset columns are in French. We will convert it to English for better readability")
    with st.expander("Renamed Columns"):
        st.table(pd.DataFrame(df_en.columns, columns=['Variables']).reset_index(drop=True))

    # ANOVA tests
    underline_header("ANOVA Tests",2)
    st.markdown("<u>ANOVA test between geographical coordinates (latitude and longitude) and hourly counting<u>", unsafe_allow_html=True)
    with st.expander("Statistical test hypothesis and results"):
        st.write("""
             H0: the geographical coordinates do not have any significant influence on hourly counting.
            \n H1: the geographical coordinates have a significant influence on hourly counting""")
        formula = 'Hourly_counting ~ Latitude + Longitude'
        model = smf.ols(formula, data=df_en).fit()
        anova_result = sm.stats.anova_lm(model, typ=2)
        st.write("ANOVA result:", anova_result)
        st.write("PR(>F) is 0.0 therefore we can reject H0, geographical variables may have a significant impact on hourly counting.")

    # Second ANOVA test
    st.markdown("<u>ANOVA test between Counter_name and hourly counting</u>", unsafe_allow_html=True)
    with st.expander("Statistical test hypothesis and results"):
        st.write("""
            H0: The counter name does not have any significant influence on hourly counting.
            \nH1: The counter name has a significant influence on hourly counting""")
        formula2 = 'Hourly_counting ~ Counter_name'
        model2 = smf.ols(formula2, data=df_en).fit()
        anova_result2 = sm.stats.anova_lm(model2, typ=2)
        st.write("ANOVA result :", anova_result2)
        st.write("PR(>F) is 0.0 therefore we can reject H0, counter name may have a significant impact on hourly counting.")

    # Spearman Test
    underline_header("Spearman Correlation Test",2)
    with st.expander("Statistical test hypothesis and results"):
        st.write("""
            H0: There is no significant correlation between hourly counting and counting hours.
            \nH1: There is a significant correlation between hourly counting and counting hours.""")
        correlation, p_value = spearmanr(df_en['counting_hour'], df_en['Hourly_counting'])
        st.write("Spearman correlation test")
        st.write("Correlation coefficient:", correlation)
        st.write("p-value:", p_value)
        st.write("""
                A correlation coefficient of 0.3238 indicates a weak positive correlation between the two variables.
                \nThe very low p-value (<0.05) indicates that this correlation is statistically significant, implying that it is unlikely to have occurred by chance. 
                \nWe can therefore reject H0.
                """)
    st.text(" ")
    underline_header("To conclude",2)
    st.write("""The variability and presence of high counts during certain hours and significant difference between median and mean could imply **the need for different operational strategies during peak times**.
             \nIt may be useful to **look into the specific hours or conditions that lead to these high counts to better understand and predict busy periods**.
             \nBy the way, we could observe through our statistical tests that the spatial variables may have an impact on hourly counting.
             \nThis information seems obvious and **we will consider geographical variable during our analysis**""")

if page == pages[2]:

    underline_header('Data visualization',2)
    
    df = load_data()
    df_en = preprocess_data(df)

    #Data Preparation for data visualisation  
    df_2023 = df_en.loc[df_en["counting_year"]==2023]
     # Group by month and hour
    df_2023_grouped = df_2023.groupby(["counting_month","counting_hour"])['Hourly_counting'].sum().reset_index()
    df_2023_grouped_2 = df_2023.groupby(["counting_day_name","counting_hour"])['Hourly_counting'].sum().reset_index()
 
    #Create a new variable for weekdays and weekends for easier ploting
    weekdays = df_2023_grouped_2[df_2023_grouped_2['counting_day_name'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
    weekends = df_2023_grouped_2[df_2023_grouped_2['counting_day_name'].isin(['Saturday', 'Sunday'])]
    # Calculate the y-axis limits so we keep the same scale
    y_min = min(weekdays['Hourly_counting'].min(), weekends['Hourly_counting'].min())
    y_max = max(weekdays['Hourly_counting'].max(), weekends['Hourly_counting'].max())
    

#Figure 1

    if st.button("Target variable distribution before data preprocesing"):
        fig = px.box(df_en, y="Hourly_counting")
        fig.update_layout(width=600)
        fig.update_layout(plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        title="Boxplot visualization of Hourly counting",
                        title_font=dict(color='black'), 
                        yaxis_title = 'Hourly counting',
                        yaxis=dict(
                            title_font=dict(color='black'), 
                            tickfont=dict(color='black')),
                        xaxis=dict(title_font=dict(color='black'),  
                            tickfont=dict(color='black')))
        st.plotly_chart(fig)
        st.markdown("""
                        ##### Main Takeaways
                        - The median number of hourly counting is 41
                        - On this graph we see a high amount of outliers
                        - Since there are a lot of outliers it's difficult to interpret the boxplot and we would need to modify the data set.
                        """)
        
#Figure 2
    # Using Boxplot to see the IQR, the mean and the outliers of the target variable
   
    if st.button("Target variable distribution after data preprocesing") :
        with st.expander("Box plot of hourly counting per month (April-Dec 2023)"):
            # Histogram of total hourly counting per month (2023)
            fig2 = px.box(df_2023_grouped, x='counting_month', y='Hourly_counting')
            fig2.update_layout(width=600)
            fig2.update_layout(plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                          yaxis_title = 'Hourly counting',
                          xaxis_title= 'Months',
                            xaxis=dict(
                                tickmode='array',
                                tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                title_font=dict(color='black'),  
                                tickfont=dict(color='black')  
                            ),
                            yaxis=dict(
                                title_font=dict(color='black'), 
                                tickfont=dict(color='black')  
                            ),)
            st.plotly_chart(fig2)
            st.markdown("""
                        ##### Main Takeaways
                        - Every month from April to December we have mostly between 80000 and 400000 bicycle counts
                        - We have still a lot of outliers that might indicate that there are some traffic points where we have much more counts than expected.
                        """)
    
#Figure 3
        with st.expander("Total number of hourly counts per month"):
             fig3= px.bar(df_2023_grouped, x='counting_month',
                         y='Hourly_counting',
                         color = 'counting_month',
                         labels={'counting_month': 'Month'})
             fig3.update_layout(
                               xaxis_title= 'Months',
                               yaxis_title = 'Hourly counting',
                               xaxis=dict(
                                   tickmode='array',
                                   tickvals=[1,2,3,4,5,6,7,8,9,10,11,12],
                                   ticktext=['Jan', 'Feb','Mar', 'Apr','May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
             st.plotly_chart(fig3)
             st.markdown("""
                        ##### Main Takeaways
                        - Hourly count per Month in 2023 to see on which months we have the most counts
                        - We can see here that the most counts are in June and September
                        - Least counts are in Aug, probably due to vacation period
                        """)
#Figure4
        with st.expander("Line plot of hourly counting per hour (2023)"):
            #Only take 2023 into account because we want the yearly trend and this year has more complete data
            fig4= px.line(df_2023_grouped, x='counting_hour',
                          y='Hourly_counting',
                          color='counting_month',
                          symbol= "counting_month",
                          labels={'counting_month': 'Month', 'counting_hour': 'Hour of Day', 'Hourly_counting': 'Hourly Counting'})
            fig4.update_layout(plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                              xaxis_title= 'Hour of Day',
                              yaxis_title = 'Hourly counting',
                              xaxis=dict(
                                  tickmode='array',
                                  tickvals=[0, 6, 7, 12, 16, 17, 18, 24],
                                  ticktext=['Midnight', '6','7', 'Noon','4','5', '6', 'Midnight'],
                                  title_font=dict(color='black'),  
                                  tickfont=dict(color='black')  
                              ),
                              yaxis=dict(
                                  title_font=dict(color='black'), 
                                  tickfont=dict(color='black')  
                              ),)
            st.plotly_chart(fig4)
            st.markdown("""
                        ##### Main Takeaways
                        - On this graph we can see that the hourly count per hour is different regarding the month 
                        - We can see that the trend is the same, since we can observe the same peaks and same lows
                        - There is a slight change then comparing June and December
                        """)


#Figure5
    # Target variable groupped by day of the week (for 2023)
        with st.expander("Hourly counting box plot per day (2023)"):
             # Define the order of days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            # Ensure the 'counting_day_name' column is in the correct order
            df_2023_grouped_2['counting_day_name'] = pd.Categorical(df_2023_grouped_2['counting_day_name'], categories=day_order, ordered=True)
    
            fig5=px.box(df_2023_grouped_2, x='counting_day_name', y='Hourly_counting', category_orders={'counting_day_name': day_order})
            fig5.update_layout(plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                               xaxis_title= 'Day of the week',
                               yaxis_title = 'Hourly counting',      
                               yaxis=dict(
                                   title_font=dict(color='black'), 
                                   tickfont=dict(color='black')),
                               xaxis=dict(
                                   title_font=dict(color='black'), 
                                   tickfont=dict(color='black')))
            st.plotly_chart(fig5)
            st.markdown("""
                        ##### Main Takeaways
                        - When we show the hourly couning per day, we can see that there is more trafic during the weekdays
                        - Also, on the weekdays seem to be more outliers
                        """)

                     
   
#Figure6
    # Create subplots with 1 row and 2 columns
        with st.expander("Hourly counting bar plot per weekday and weekend (2023)"):
            fig6a= px.bar(weekdays, x='counting_hour', y='Hourly_counting', color='counting_day_name', barmode='group',
                          labels={'counting_day_name': 'Weekday'})
            fig6a.update_layout(
                title="Hourly counting on weekdays",
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis_title='Weekday',
                yaxis_title='Hourly counting',
                yaxis=dict(
                    range=[y_min, y_max],
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')),
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 6, 7, 12, 16, 17, 18, 24],
                    ticktext=['Midnight', '6','7', 'Noon','4','5', '6', 'Midnight'], 
                    title_font=dict(color='black'), 
                    tickfont=dict(color='black')),
                title_font=dict(color='black') )
            st.plotly_chart(fig6a)
            st.markdown("""
                        ##### Main Takeaways
                        - We can observe a peak in hourly counting during rush hours (in the morning from 6 to 7 and during the afternoon from 16 to 17) 
                        - We can conclude that a lot of workers / students are using bicycles on a daily basis during working hours
                        """)
            
    
            # Boxplot for the weekends
            fig6b= px.bar(weekends, x='counting_hour', y='Hourly_counting', color='counting_day_name', barmode='group',
                          labels={'counting_day_name': 'Weekend'})
            fig6b.update_layout(title="Hourly counting on weekend",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis_title='Weekday',
                    yaxis_title='Hourly counting',
                    yaxis=dict(
                        range=[y_min, y_max],
                        title_font=dict(color='black'),
                        tickfont=dict(color='black')),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[0, 6, 7, 12, 16, 17, 18, 24],
                        ticktext=['Midnight', '6','7', 'Noon','4','5', '6', 'Midnight'], 
                        title_font=dict(color='black'), 
                        tickfont=dict(color='black')),
                    title_font=dict(color='black') )
            st.plotly_chart(fig6b)
            st.markdown("""
                        ##### Main Takeaways
                        - On a weekend the rush hours change, shifting to the afternoon 
                        - We don't have the same peaks and the curve is more flat.
                        """)

#Figure7
        # Creating a heatmap to observe the counting numbers related to temporal data (day / hour)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Create a pivot table
        pivot_table = df_2023.pivot_table(index='counting_hour', columns='counting_day_name', values='Hourly_counting', aggfunc='median')
        pivot_table = pivot_table[day_order]
        # Plotting the heatmap
        with st.expander("Heatmap of hourly counting by day / hour"):
                  fig7 = go.Figure(data=go.Heatmap(
                            z=pivot_table.values,
                            x=pivot_table.columns,
                            y=pivot_table.index,
                            colorscale='RdYlBu_r'
                        ))
                  fig7.update_layout(plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                     paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                     xaxis_title= 'Day',
                                     yaxis_title = 'Hour of the day',
                                     yaxis=dict(
                                            tickmode='array',
                                            tickvals=[0, 6, 7, 12, 16, 17, 18, 24],
                                            ticktext=['Midnight', '6','7', 'Noon','4','5', '6', 'Midnight'],
                                            title_font=dict(color='black'), 
                                            tickfont=dict(color='black')),
                                    xaxis=dict(title_font=dict(color='black'), 
                                          tickfont=dict(color='black'),))
                  st.plotly_chart(fig7)
                  st.markdown("""
                                    ##### Main Takeaways
                                    - We can observe that the rush hours involve more crowded bicycle counting with a sharp decrease during weekends.
                                    """)

#Figure8


# Identify the top 3 counters
 
        #Total counts per counter
        df_2023_grouped_counter_total = df_2023.groupby(["Counter_ID"])['Hourly_counting'].sum().reset_index().sort_values(by='Hourly_counting', ascending=False)
        
        # Group by counter and hour
        df_2023_grouped_counter = df_2023.groupby(["Counter_ID", 'counting_hour'])['Hourly_counting'].sum().reset_index().sort_values(by='Hourly_counting', ascending=False)
        
    
        # Calculate total hourly counting for all counters
        total_hourly_counting = df_2023_grouped_counter_total['Hourly_counting'].sum()
        
        # Calculate hourly counting for each counter
        hourly_counting_per_counter = df_2023_grouped_counter_total.groupby('Counter_ID')['Hourly_counting'].sum()
        
        # Sort counters by hourly counting descending
        hourly_counting_per_counter = hourly_counting_per_counter.sort_values(ascending=False)
        
        # Separate top 10 counters, lowest 10 counters, and others
        top10_counters = hourly_counting_per_counter.head(10)
        lowest10_counters = hourly_counting_per_counter.tail(10)
        other_counters = hourly_counting_per_counter[10:-10]  # All counters except top 10 and lowest 10
        
        # Calculate total hourly counting for top 10 and lowest 10 counters
        top10_total = top10_counters.sum()
        lowest10_total = lowest10_counters.sum()
        
        # Calculate percentage for top 10 counters, lowest 10 counters, and 'Others'
        top10_percentages = (top10_total / total_hourly_counting) * 100
        lowest10_percentages = (lowest10_total / total_hourly_counting) * 100
        others_percentage = ((total_hourly_counting - top10_total - lowest10_total) / total_hourly_counting) * 100
        
        # Create labels and values for the Pie chart
        labels = ['Top 10', 'Lowest 10', 'Others']
        values = [top10_percentages, lowest10_percentages, others_percentage]
        with st.expander("Share of Top 10 & Flop 10 counters among the hourly counting total"):
                # Create a Pie chart
                fig8 = go.Figure(data=[go.Pie(labels=labels, values=values)])
                
                # Update layout with title and legend
                fig8.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    showlegend=True,
                    legend=dict(orientation="v",yanchor="top",
                        y=0.9, xanchor="right", x=0.9, bgcolor='white',font=dict(color='black')))
                st.plotly_chart(fig8)


                unique_counts = df_2023.drop_duplicates(subset=['Counter_ID'])['Counter_ID'].value_counts()
            
                unique_counts = df_2023['Counter_ID'].drop_duplicates().count()
                
                # Display the sum of unique counts
                st.markdown(f"""
                    ##### Main Takeaways
                    - We can observe that the top 10 counters make up to a quarter of all counters
                    - In Paris, we have {unique_counts} unique counters
                """)


#Figure9

        # Keeping only the necessary data
        df_map = df_2023[['Counter_site_name', 'Hourly_counting', 'Latitude', 'Longitude']]
        
        # Calculate the sum of hourly_counting for each counter
        counter_sum = df_map.groupby('Counter_site_name')['Hourly_counting'].sum().reset_index()
        
        # Identify the top 10 counters with the maximum sum of hourly counting
        with st.expander("Top 10 Counters (in hourly counting total)"):
                top_10_counters_en = counter_sum.nlargest(10, 'Hourly_counting')
                st.dataframe(top_10_counters_en)
            
            # Identify the flop 10 counters with the least sum of hourly counting
        with st.expander("Flop 10 Counters (in hourly counting total)"):
                flop_10_counters_en = counter_sum.nsmallest(10, 'Hourly_counting')
                st.dataframe(flop_10_counters_en)
                
        # Fusionner avec le DataFrame principal pour obtenir les Latitude et Longitude
        top_10 = pd.merge(top_10_counters_en, df_map[['Counter_site_name', 'Latitude', 'Longitude']].drop_duplicates(), on='Counter_site_name')
        flop_10 = pd.merge(flop_10_counters_en, df_map[['Counter_site_name', 'Latitude', 'Longitude']].drop_duplicates(), on='Counter_site_name')
        
        # Combiner les top 10 et flop 10 dans un seul DataFrame
        df_top_flop = pd.concat([top_10, flop_10]).reset_index(drop=True)
        
        with st.expander("Location of Top 10 and Flop 10 counters in Paris")  :
                # Créer une figure initiale vide
                fig9 = px.scatter_mapbox(lat=[], lon=[], zoom=12, height=600)
                
                # Ajouter les points pour le top 10 en vert
                fig9.add_trace(px.scatter_mapbox(top_10, lat='Latitude', lon='Longitude', hover_name='Counter_site_name', size =top_10['Hourly_counting'], 
                                                color_discrete_sequence=['green']).data[0])
                
                # Ajouter les points pour le flop 10 en rouge
                fig9.add_trace(px.scatter_mapbox(flop_10, lat='Latitude', lon='Longitude', hover_name='Counter_site_name',
                                                color_discrete_sequence=['red']).data[0])
                
                # Mettre à jour la mise en page pour une meilleure visualisation et se concentrer sur Paris intra muros
                fig9.update_layout(mapbox_style="open-street-map", plot_bgcolor= 'rgba(0, 0, 0, 0)', paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                  mapbox=dict(center=dict(lat=48.8566, lon=2.3522), zoom=10))
                
                # Définir une taille minimale pour les points
                fig9.update_traces(marker=dict(sizemin=10, sizemode='area'))
                
                # Afficher le graphique
                st.plotly_chart(fig9) 

if page == pages[3]:
    df = load_data()
    df_en = preprocess_data(df)
    underline_header('Data preparation',2)
    with st.expander("The dataset"):
         df.info()
         
    st.write("4 main steps were implemented in order to process data and prepare it for machine leaning models :")
    with st.expander("**Null values management**"):
        missing_counts = df_en.isna().sum()
        missing_counts = missing_counts
        missing_values_summary = pd.DataFrame({
        'Variable': missing_counts.index,
        'Number of missing values': missing_counts.values
        }).reset_index(drop=True)
        st.table(missing_values_summary)
        st.text(" ")
        st.write(""" 
                 We can observe that the most impacted columns regarding missing values are not the most important for our project since they are pictures or URL data. 
                 We also observe that missing values seem to miss geographical datas and installation data information.
                 They represent **only 1,6 %** of the total dataset and it seems relevant to delete these entries since they could skew our analysis if we are unable to locate these counters""")
    with st.expander("**Outliers management**"):
        st.write("""Assuming we apply the classic formula to identify the outliers **(Q1 - 1.5 * IQR and  Q3 + 1.5 * IQR)** we find 79 981 values.
                 This calculation suggests that any hourly counting > 217 is an outlier, which is not relevant.""")
        st.text(" ")
        st.write("""These values seem to be extreme values and we will not consider these values as outliers. We need to pay attention to every value to assess its relevance.
                 Doing this we observe that the majority of extreme outliers are counted by one counter located in central Paris, near the Luxembourg Gardens, during the summer season.
                 This area is generally busy due to its central location and the presence of the garden, which is a tourist attraction.""")
        st.text(" ")

        st.write("""It is likely that this area sees a significant amount of traffic, especially from tourists, students from the University of Sorbonne, and locals.
                **Therefore we could keep these values and delete the other extreme values of the entrie 501 925 (8 190 hourly counts)**  """)
    with st.expander("**Feature engineering**"):
        st.write("As a reminder, the current dataframe :")
        st.table(pd.DataFrame(df_en.columns, columns=['Variables']).reset_index(drop=True))
        st.write(""" After having splitted the temporal data and having made small changes on the columns we will **keep only the relevant numerical / categorical values**.
                 This latter will be encoded and we will get rid of the images and URL.
                 Here we present the updated dataframe we will use to train our models :""")
        df_en[['Counter_ID1', 'Counter_ID2']] = df_en['Counter_ID'].str.split('-', expand=True)
        df_en['Installation_date_of_the_counting_site'] = pd.to_datetime(df_en['Installation_date_of_the_counting_site'])
        df_en['installation_year'] = df_en['Installation_date_of_the_counting_site'].dt.year
        df_en['installation_month'] = df_en['Installation_date_of_the_counting_site'].dt.month
        df_en['installation_day'] = df_en['Installation_date_of_the_counting_site'].dt.day
        df_processed = df_en[["Counter_ID1", "Counter_ID2","Counter_site_name", "Hourly_counting","Latitude","Longitude","counting_year", "counting_month","counting_day","counting_hour","counting_day_name","weekday","weekend","installation_year","installation_month","installation_day"]]
        st.table(pd.DataFrame(df_processed.columns, columns=['Variables']).reset_index(drop=True))
    with st.expander("**Encoding data**"):
        st.write("The only categorical columns that we need to encode are the **counting_day_name** and the **Counter_site_name**. We use the **factorize encoding method** to achieve this.")
       
if page == pages[4]:
    underline_header('Modelling',2)
    st.write('After having implemented the processing part, we are able to train some machine learning models in order to provide accurate predictions to Paris authorities')
    with st.expander("The performance metrics of the 6 trained models"):
        data = {
    "Model": ["Random Forest", "Lasso", "Decision Tree", "Ridge", "Linear", "Lasso CV"],
    "MAE Train set": [5.99, 63.7, 31.9, 63.9, 63.7, 63.7],
    "MAE Test set": [14.623, 63.82, 31.9, 63.5, 63.8, 63.8],
    "MSE Train set": [222.18, 9758.9, 3664.2, 9787.4, 9760.8, 9761.4],
    "MSE Test set": [1099.28, 9869.3, 3786.3, 9761.7, 9841.3, 9842.1],
    "RMSE Train set": [14.5, 99.78, 60.5, 98.9, 98.7, 98.8],
    "RMSE Test set": [35.42, 99.34, 61.5, 98.8, 99.2, 99.2],
    "R² Train set": [0.98, 0.11, 0.67, 0.11, 0.11, 0.11],
    "R² Test set": [0.90, 0.11, 0.65, 0.11, 0.11, 0.11]}
        # Create the DataFrame
        model_results = pd.DataFrame(data)
        # Display the DataFrame in Streamlit
                # Create HTML for the table with black text and bold for the first row
        table_html = """
        <table style='width:100%; border-collapse: collapse;'>
            <thead>
                <tr style='border-bottom: 1px solid black;'>
                    <th>Model</th>
                    <th>MAE Train set</th>
                    <th>MAE Test set</th>
                    <th>MSE Train set</th>
                    <th>MSE Test set</th>
                    <th>RMSE Train set</th>
                    <th>RMSE Test set</th>
                    <th>R² Train set</th>
                    <th>R² Test set</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, row in model_results.iterrows():
            style = "font-weight: bold; background-color: white;"  if i == 0 else ""
            table_html += f"<tr style='border-bottom: 1px solid black; {style}'>"
            for item in row:
                table_html += f"<td style='padding: 8px; text-align: left; color: black;'>{item}</td>"
            table_html += "</tr>"
        
        table_html += """
            </tbody>
        </table>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)        
      
        st.write("""Lasso, Ridge, Linear, and Lasso CV models have very similar performance metrics with significantly higher error rates (MAE, MSE, RMSE) and much lower R² values around **0.11 for both train and test sets**.
                 Decision Tree shows better performance than Lasso and Ridge models but still has significantly higher error rates than Random Forest.
                 **The Random Forest model has the lowest MAE, MSE, and RMSE on the test set compared to all other models**. This indicates that the predictions made by the Random Forest are, on average, closer to the actual values than those made by the other models.
                 **The R² value for the Random Forest is 0.90** on the test set, which is substantially higher than those of the other models . It  indicates a better fit of the model to the data.
                 """)
    st.text(" ")
    # Display the plots of predictions vs actual values & feature importance
    underline_header('Scatter plot of RF model predicted values compared to actual values',2)
    st.text(" ")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'RF_scatter_pred.jpg')
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.error(f"Image {image_path} not found.")
    st.text(" ")
    underline_header('Features importance',2)
    st.text(" ")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'RF_feat_imp.jpg')
    if os.path.exists(image_path):
        st.image(image_path, caption='Top 5 features')
    else:
        st.error(f"Image {image_path} not found.")
    st.write("""Although the Random Forest model shows some increase in error from the training set to the test set, its performance degradation is much less severe than that of the Decision Tree. This suggests that the Random Forest model generalizes better to unseen data.
Given these points, **the Random Forest model demonstrates a strong ability to predict the target variable accurately and with less error compared to the other models.** 
This makes it a relevant choice for your predictions.""")
    st.text(" ")
    st.write("As a final step before predictions we will only keep the most importance feature according to the plot above in order to optimize our model's efficiency")

if page == pages[5]:
    underline_header('Predictions',2)

    df = load_data()
    df_en = preprocess_data(df)

    # Widgets for selecting a date and a time
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input('Select a date', value=pd.to_datetime('2023-01-01'))
    with col2:
        selected_time = st.time_input('Select an hour', value=pd.to_datetime('10:00').time())

    # Combine the selected date and time into a datetime
    selected_datetime = pd.to_datetime(f"{selected_date} {selected_time}")

    # Widget for selecting a Counter_site_name
    selected_counter_site_name = st.selectbox('Select a Counter Site Name', df_en['Counter_site_name'].unique())

    if st.button('Predict'):
        # Prepare the input data for the model
        selected_row = df_en[df_en['Counter_site_name'] == selected_counter_site_name].iloc[0]
        
        input_data = pd.DataFrame({
            'Counter_site_name': [selected_counter_site_name],
            'Latitude': [selected_row['Latitude']],
            'Longitude': [selected_row['Longitude']],
            'counting_year': [selected_datetime.year],
            'counting_month': [selected_datetime.month],
            'counting_day': [selected_datetime.day],
            'counting_hour': [selected_datetime.hour],
            'counting_day_name': [selected_datetime.strftime('%A')],
            'weekday': [1 if selected_datetime.weekday() < 5 else 0],
            'weekend': [1 if selected_datetime.weekday() >= 5 else 0]
        })

        # Ensure the order of columns matches the one used during training
        expected_columns = ['Counter_site_name', 'Latitude', 'Longitude', 'counting_year', 
                            'counting_month', 'counting_day', 'counting_hour', 
                            'counting_day_name', 'weekday', 'weekend']
        input_data = input_data[expected_columns]

        # Convert categorical columns to dummy variables
        input_data = pd.get_dummies(input_data, columns=['Counter_site_name', 'counting_day_name'])

        # Make prediction if the model is loaded
        if rf_model is not None:
            # Ensure all columns used during training are present
            for col in rf_model.feature_names_in_:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Reorder columns to match training order
            input_data = input_data[rf_model.feature_names_in_]
            
            prediction = rf_model.predict(input_data)
            st.success(f'Predicted counting for {selected_counter_site_name} on {selected_datetime}: {prediction[0]:.0f} bicyles should be counted')
        else:
            st.error("Model not loaded, prediction impossible.")

       # Optionally display a historical data graph for the selected counter
    if st.checkbox('Show historical data for the selected counter'):
        historical_data = df_en[df_en['Counter_site_name'] == selected_counter_site_name]

        # Ensure Counting_date_hour is in datetime format
        if 'Counting_date_hour' not in historical_data.columns:
            st.error("Column 'Counting_date_hour' not found in the data.")
        else:
            if not pd.api.types.is_datetime64_any_dtype(historical_data['Counting_date_hour']):
                try:
                    historical_data['Counting_date_hour'] = pd.to_datetime(historical_data['Counting_date_hour'])
                except Exception as e:
                    st.error(f"Error converting 'Counting_date_hour' to datetime: {e}")
            
            if 'Hourly_counting' not in historical_data.columns:
                st.error("Column 'Hourly_counting' not found in the data.")
            else:
                if historical_data.empty:
                    st.warning("No historical data available for the selected counter.")
                else:
                    # Group data by day
                    historical_data['Counting_date'] = historical_data['Counting_date_hour'].dt.date
                    daily_data = historical_data.groupby('Counting_date').agg({'Hourly_counting': 'sum'}).reset_index()

                    fig = px.line(daily_data, x='Counting_date', y='Hourly_counting', 
                                  title=f'Historical data for {selected_counter_site_name}', 
                                  labels={'Counting_date': 'Date', 'Hourly_counting': 'Total Daily Counting'},
                                  template='plotly_white')
                    
                    fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    })
                    
                    st.plotly_chart(fig)
        
        # Optionally display the counter site on a map
    if st.checkbox('Show counter site'):
        selected_counter_data = df_en[df_en['Counter_site_name'] == selected_counter_site_name].iloc[0]
        map_data = pd.DataFrame({
            'Latitude': [selected_counter_data['Latitude']],
            'Longitude': [selected_counter_data['Longitude']],
            'Counter_site_name': [selected_counter_site_name]
        })
    
        fig = px.scatter_mapbox(map_data, lat='Latitude', lon='Longitude', text='Counter_site_name',
                                zoom=12, height=400)
        fig.update_traces(marker=dict(size=20, color='red', opacity=0.8))
        fig.update_layout(mapbox_style="open-street-map", 
                        margin={"r":0,"t":0,"l":0,"b":0},
                        mapbox=dict(center=dict(lat=48.8566, lon=2.3522)))
        st.plotly_chart(fig)
