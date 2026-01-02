import pandas as pd
import re

COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
    'state_info', 'party_affiliation', 'barely_true_counts', 
    'false_counts', 'half_true_counts', 'mostly_true_counts', 
    'pants_on_fire_counts', 'context'
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def load_and_process_data(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None, names=COLUMNS)
    
    df['clean_statement'] = df['statement'].apply(clean_text)
    
    df['combined_text'] = (
        df['party_affiliation'].fillna('none') + " " + 
        df['speaker'].fillna('none') + " " + 
        df['subject'].fillna('none') + " " + 
        df['context'].fillna('none') + " " + 
        df['clean_statement']
    )
    
    fake_labels = ['pants-fire', 'false', 'barely-true']
    df['target'] = df['label'].apply(lambda x: 1 if x in fake_labels else 0)
    
    return df