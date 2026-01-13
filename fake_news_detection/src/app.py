import streamlit as st
import mlflow.sklearn
import pandas as pd
import re

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♀️", layout="wide")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def construct_input(party, speaker, subject, context, statement):
    return f"{party} {speaker} {subject} {context} {clean_text(statement)}".lower()

@st.cache_data
def load_test_data():
    try:
        cols = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'ctx']
        return pd.read_csv("data/test.tsv", sep='\t', header=None, names=cols)
    except: return pd.DataFrame()

test_df = load_test_data()
st.sidebar.header("Configuration")
run_id = st.sidebar.text_input("MLflow Run ID:")

if st.sidebar.button("🎲 Random Test Case"):
    if not test_df.empty:
        row = test_df.sample(1).iloc[0]
        st.session_state['speaker'] = str(row['speaker'])
        st.session_state['party'] = str(row['party'])
        st.session_state['subject'] = str(row['subject']) 
        st.session_state['context'] = str(row['ctx'])     
        st.session_state['statement'] = str(row['statement'])
        st.session_state['actual_label'] = str(row['label'])

if run_id:
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        st.sidebar.success("Model Loaded!")
    except: st.sidebar.error("Invalid Run ID")
else:
    st.stop()

st.title("🕵️‍♀️ Fake News Detector (Pro)")

if 'actual_label' in st.session_state:
    lbl = st.session_state['actual_label']
    st.info(f"📝 Actual Label: **{lbl.upper()}**")

with st.form("prediction_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1: speaker = st.text_input("Speaker", value=st.session_state.get('speaker', 'donald-trump'))
    with c2: party = st.text_input("Party", value=st.session_state.get('party', 'republican'))
    with c3: subject = st.text_input("Subject", value=st.session_state.get('subject', 'economy'))
    with c4: context = st.text_input("Context", value=st.session_state.get('context', 'tweet'))
    
    statement = st.text_area("Statement", value=st.session_state.get('statement', 'Says...'))
    submit = st.form_submit_button("Check Veracity")

if submit:
    final_input = construct_input(party, speaker, subject, context, statement)
    input_data = pd.DataFrame({
        'combined_text': [final_input],
        'barely_true_counts': [0],
        'false_counts': [0],
        'half_true_counts': [0],
        'mostly_true_counts': [0],
        'pants_on_fire_counts': [0]
    })
    try:
        probs = model.predict_proba(input_data)[0]
        if probs[1] > 0.5:
            st.error(f"🚨 FAKE ({probs[1]:.1%} Confidence)")
            st.progress(float(probs[1]))
        else:
            st.success(f"✅ REAL ({probs[0]:.1%} Confidence)")
            st.progress(float(probs[0]))
            
        with st.expander("Debug Info"):
            st.write(f"**Processed Text:** `{final_input}`")
            st.write("**Full DataFrame Input:**")
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"Model Error: {e}")