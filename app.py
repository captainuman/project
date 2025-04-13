import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import scipy
from sklearn.cluster import KMeans
import os
print("Current working directory:", os.getcwd())

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

df = preprocessor.preprocess(df,region_df)
st.sidebar.title('Olympics Analysis')
st.sidebar.image("C:\project\download.jpeg")

user_menu = st.sidebar.radio(
    'select an option',("Olympics_Analysis","Olympic_Pridiction"))

if user_menu == "Olympics_Analysis":
    user_menu = st.sidebar.radio(
        'select an option',
        ('Medal Tally', 'Overall-Analysis', 'Country-wise Analysis', 'Athlete-wise Analysis')
    )

    if user_menu == 'Medal Tally':
        st.sidebar.header('Medal Tally')
        years, country = helper.country_year_list(df)

        selected_year = st.sidebar.selectbox("Select Year", years)
        selected_country = st.sidebar.selectbox("Select Country", country)

        # Fetch medal tally
        medal_tally_df = helper.fetch_medal_tally(df, selected_year, selected_country)

        if selected_year == 'Overall' and selected_country == 'Overall':
            st.title('Overall Tally')
        if selected_year != 'Overall' and selected_country == 'Overall':
            st.title(f'Medal Tally in {str(selected_year)} Olympics')
        if selected_year == 'Overall' and selected_country != 'Overall':
            st.title(f'{selected_country} Overall Performance')
        if selected_year != 'Overall' and selected_country != 'Overall':
            st.title(f'{selected_country} Performance in {str(selected_year)} Olympics')

        st.table(medal_tally_df)

        # Apply K-Means clustering to the medal tally data
        kmeans = KMeans(n_clusters=4, random_state=42) # You can change the number of clusters (k) as needed
        medal_tally_df['Cluster'] = kmeans.fit_predict(medal_tally_df[['Gold', 'Silver', 'Bronze', 'total']])

        # Visualizing the clustering on a scatter plot (using Gold vs Silver as an example)
        fig = px.scatter(medal_tally_df, x='Gold', y='Silver', color='Cluster',
                         title="Medal Tally Clustering (K-Means)", hover_data=['region', 'Bronze', 'total'])
        st.plotly_chart(fig)

        # Optionally, you can also display the cluster centers (centroids) of the countries
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Gold', 'Silver', 'Bronze', 'total'])
        centroids['Cluster'] = range(4)
        st.write("Cluster Centers (Centroids):")
        st.table(centroids)

    if user_menu == 'Overall-Analysis':
        editions = df['Year'].unique().shape[0] - 1
        cities = df['City'].unique().shape[0]
        sports = df['Sport'].unique().shape[0]
        events = df['Event'].unique().shape[0]
        athletes = df['Name'].unique().shape[0]
        nations = df['region'].unique().shape[0]

        st.title('Top Statistics')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Editions")
            st.title(editions)
        with col2:
            st.header("Hosts")
            st.title(cities)
        with col3:
            st.header("Sports")
            st.title(sports)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Events")
            st.title(events)
        with col2:
            st.header("Nations")
            st.title(nations)
        with col3:
            st.header("Athletes")
            st.title(athletes)

        nations_over_time = helper.data_over_time(df, 'region')
        fig = px.line(nations_over_time, x='Edition', y='region')
        st.title('\n Participating Nations over the years')
        st.plotly_chart(fig)

        events_over_time = helper.data_over_time(df, 'Event')
        fig = px.line(events_over_time, x='Edition', y='Event')
        st.title('\n Events over the years')
        st.plotly_chart(fig)

        athletes_over_time = helper.data_over_time(df, 'Name')
        fig = px.line(athletes_over_time, x='Edition', y='Name')
        st.title('\n Athletes over the years')
        st.plotly_chart(fig)

        st.title('No. of Events over time(Every sport)')
        fig, ax = plt.subplots(figsize=(20, 20))
        x = df.drop_duplicates(['Year', 'Sport', 'Event'])
        ax = sns.heatmap(
            x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
            annot=True)
        st.pyplot(fig)

        st.title('Most successful Athletes')
        sport_list = df['Sport'].unique().tolist()
        sport_list.sort()
        sport_list.insert(0, 'Overall')

        selected_sport = st.selectbox('Select a sport', sport_list)
        x = helper.most_successful(df, selected_sport)
        st.table(x)

    if user_menu == 'Country-wise Analysis':
        st.sidebar.title('Country-wise Analysis')

        country_list = df['region'].dropna().unique().tolist()
        country_list.sort()
        selected_country = st.sidebar.selectbox('Select a country', country_list)

        country_df = helper.yearwise_medal_tally(df, selected_country)
        fig = px.line(country_df, x='Year', y='Medal')
        st.title(selected_country + 'Medal over the years')
        st.plotly_chart(fig)

        st.title(selected_country + ' excels in the following sports')
        pt = helper.country_event_heatmap(df, selected_country)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax = sns.heatmap(pt, annot=True)
        st.pyplot(fig)

        st.title('top 10 athletes of ' + selected_country)
        top10_df = helper.most_successful_countrywise(df, selected_country)
        st.table(top10_df)

    if user_menu == 'Athlete-wise Analysis':
        athlete_df = df.drop_duplicates(subset=['Name', 'region'])

        x1 = athlete_df['Age'].dropna()
        x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
        x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
        x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

        # Display the age distribution based on medals
        fig = ff.create_distplot([x1, x2, x3, x4],
                                 ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze medalist'],
                                 show_hist=False)
        fig.update_layout(autosize=False, width=1000, height=600)
        st.title('Distribution of Age')
        st.plotly_chart(fig)

        x = []
        name = []
        famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                         'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                         'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                         'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                         'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                         'Tennis', 'Golf', 'Softball', 'Archery',
                         'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                         'Rhythmic Gymnastics', 'Rugby Sevens',
                         'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo',
                         'Ice Hockey']

        for sport in famous_sports:
            temp_df = athlete_df[athlete_df['Sport'] == sport]
            x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
            name.append(sport)

        # Display age distribution by sport
        fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
        fig.update_layout(autosize=False, width=1000, height=600)
        st.title('Distribution of Age wrt Sports')
        st.plotly_chart(fig)

        sport_list = df['Sport'].unique().tolist()
        sport_list.sort()
        sport_list.insert(0, 'Overall')

        selected_sport = st.selectbox('Select a sport', sport_list)
        temp_df = helper.weight_v_height(df, selected_sport)
        fig, ax = plt.subplots()
        ax = sns.scatterplot(x=temp_df['Weight'], y=temp_df['Height'], hue=temp_df['Medal'], style=temp_df['Sex'], s=60)
        st.pyplot(fig)

        # Add K-Means clustering for athlete data (Age, Height, Weight)
        st.title('Athlete Clustering (K-Means)')
        athlete_clusters, model = helper.run_kmeans_on_athletes(df, k=4)

        # Display the dataframe with clusters (optional)
        st.dataframe(athlete_clusters.head())

        # Plotting K-Means clusters on a scatter plot (Height vs. Weight)
        fig = px.scatter(athlete_clusters, x='Height', y='Weight', color=athlete_clusters['Cluster'].astype(str),
                         title="Athlete Clustering (K-Means)", hover_data=['Age'])
        st.plotly_chart(fig)

        st.title('Men vs Women Participation Over the Years')
        final = helper.men_vs_women(df)
        fig = px.line(final, x='Year', y=['Male', 'Female'])
        st.plotly_chart(fig)

if user_menu == 'Olympic_Pridiction':
    user_menu = st.sidebar.radio(
        'select an option',
        ('Medal Prediction','Country Medal Prediction')
    )
if user_menu == 'Medal Prediction':
    st.title("🎯 Athlete Medal Prediction")
    st.write("Select an athlete or fill in custom details to predict their medal chances.")

    use_existing = st.radio("Choose Input Method:", ["Select Athlete", "Manual Entry"])

    if use_existing == "Select Athlete":
        athlete_names = df.dropna(subset=['Name', 'Age', 'Weight', 'Height', 'Sport', 'Event', 'Sex'])[
            'Name'].unique().tolist()
        athlete_names.sort()
        selected_athlete = st.selectbox("Select Athlete", athlete_names)

        athlete_row = \
        df[df['Name'] == selected_athlete].dropna(subset=['Age', 'Weight', 'Height', 'Sport', 'Event', 'Sex']).iloc[0]

        age = int(athlete_row['Age'])
        height = int(athlete_row['Height'])
        weight = int(athlete_row['Weight'])
        sex = athlete_row['Sex']
        sport = athlete_row['Sport']
        event = athlete_row['Event']

        st.markdown(f"**Pre-filled Details for {selected_athlete}:**")
        st.write(
            f"**Age**: {age}, **Height**: {height} cm, **Weight**: {weight} kg, **Sex**: {sex}, **Sport**: {sport}, **Event**: {event}")

    else:
        age = st.number_input("Age", 10, 60)
        height = st.number_input("Height (cm)", 100, 250)
        weight = st.number_input("Weight (kg)", 30, 200)
        sex = st.selectbox("Sex", ['M', 'F'])
        sport = st.selectbox("Sport", sorted(df['Sport'].dropna().unique()))
        event = st.selectbox("Event", sorted(df[df['Sport'] == sport]['Event'].dropna().unique()))

    if st.button("Predict Medal"):
        from helper import train_medal_prediction_model

        model, le_sport, le_event, le_sex, le_medal, acc = train_medal_prediction_model(df)

        try:
            input_df = pd.DataFrame([[
                age,
                height,
                weight,
                le_sex.transform([sex])[0],
                le_sport.transform([sport])[0],
                le_event.transform([event])[0]
            ]], columns=['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event'])

            prediction = model.predict(input_df)[0]
            medal = le_medal.inverse_transform([prediction])[0]

            st.success(f"🏅 Predicted Medal: **{medal}**")
            st.info(f"📊 Model Accuracy: {acc:.2%}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

if user_menu == 'Country Medal Prediction':
    st.title("📈 Country & Sport Medal Prediction")
    st.write("Predict the number of medals a country might win in a selected **sport** for a future Olympics.")

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()
    selected_country = st.selectbox("Select Country", country_list)

    sport_list = df['Sport'].dropna().unique().tolist()
    sport_list.sort()
    selected_sport = st.selectbox("Select Sport", sport_list)

    future_year = st.number_input("Enter Future Olympic Year (e.g. 2028)", min_value=2024, max_value=2100, step=4)

    if st.button("Predict Medals"):
        from helper import train_country_sport_medal_model

        model, medal_df, r2 = train_country_sport_medal_model(df, selected_country, selected_sport)

        if model is None:
            st.warning(f"❌ Not enough data to predict medals for {selected_country} in {selected_sport}.")
        else:
            prediction = model.predict([[future_year]])[0]
            st.success(
                f"🏅 Predicted Total Medals for **{selected_country}** in **{selected_sport}** ({future_year}): **{int(prediction)}**")
            st.info(f"📊 Model R² Score: {r2:.2f}")

            # Plot history + prediction
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(medal_df['Year'], medal_df['Total_Medals'], marker='o', label='Historical')
            ax.scatter(future_year, prediction, color='red', s=100, label='Prediction')
            ax.set_xlabel("Year")
            ax.set_ylabel("Medals Won")
            ax.set_title(f"{selected_country} - {selected_sport}")
            ax.legend()
            st.pyplot(fig)