import json
import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.clustering import load_model, predict_model

DATA = "welcome_survey_simple_v2.csv"
MODEL_NAME = "welcome_survey_clustering_pipeline_v2"
CLUSTER_NAMES_AND_DESCRIPTIONS = "welcome_survey_cluster_names_and_descriptions_v2.json"

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_all_participants():
    model = get_model()
    all_data_df = pd.read_csv(DATA, sep = ";")
    return predict_model(model, data = all_data_df)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r") as f:
        return json.loads(f.read())

with st.sidebar:
    st.header("Tell us something about you.")
    st.markdown("We will help you to find people with interests similar to yours.")
    age = st.selectbox("Age:", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Educational level:", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Favourite animal:", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Favourite place:", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Gender", ['Mężczyzna', 'Kobieta'])


person_df = pd.DataFrame([
    {
        "age": age,
        "edu_level": edu_level,
        "fav_animals": fav_animals,
        "fav_place": fav_place,
        "gender": gender
    }
])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions();

predicted_cluster_id = predict_model(model, data = person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]
st.header(f"Your are closest to group: \"{predicted_cluster_data['name']}\"")
st.markdown(predicted_cluster_data["description"])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Number of your friends:", len(same_cluster_df))

st.header("People from group:")
fig = px.histogram(same_cluster_df.sort_values("age"), x = "age")
fig.update_layout(
    title = "Age distrubution in group:",
    xaxis_title = "Age",
    yaxis_title = "Number of people"
)

st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Educational level distribution in group",
    xaxis_title="Educational level",
    yaxis_title="Number of people",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Favourite animal distribution",
    xaxis_title="Favourite animals",
    yaxis_title="Number of people",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Favourite place distribution",
    xaxis_title="Favourite animals",
    yaxis_title="Number of people",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Gender distribution in group",
    xaxis_title="Gender",
    yaxis_title="Number of people",
)
st.plotly_chart(fig)

