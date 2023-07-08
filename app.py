import re
import nltk
nltk.download('all')
import keras
import spacy
import string 
import pickle
import tempfile
import numpy as np
import gradio as gr
import contractions
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Attention(Layer):
    
    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
    


def load_tokenizer(path):
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def cleaning(text):
    # Punctuation symbols to remove
    exclude = string.punctuation
    
    def expand_contractions(text): return contractions.fix(text)
    text = expand_contractions(text)

    text = text.lower()
    
    def remove_tags(text): return re.sub(r'@\w*', ' ' , text)
    text = remove_tags(text)
    
    def remove_hashtags(text): return re.sub(r'#\w*', ' ' , text)
    text = remove_hashtags(text)
    
    def remove_apostrophe(text): return re.sub(r"'s\b", "", text)
    text = remove_apostrophe(text)

    def remove_special_chars(text): return re.sub(r"[^a-zA-Z0-9\s]", ' ', text)
    text = remove_special_chars(text)
    
    def remove_number(text): return re.sub(r'[\d]', ' ', text)
    text = remove_number(text)
    
    def remove_punc(text): return ''.join([c for c in text if c not in exclude])
    text = remove_punc(text)
    
    def remove_extra_spaces(text): return re.sub('^\S', ' ', text)
    text = remove_extra_spaces(text)
    
    def map_pos_tags(pos_tags):
    # Map NLTK POS tags to WordNet tags
        tag_map = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }
        
        mapped_tags = []
        for token, tag in pos_tags:
            mapped_tag = tag[0].upper()
            if mapped_tag in tag_map:
                mapped_tag = tag_map[mapped_tag]
            else:
                mapped_tag = wordnet.NOUN  # Default to noun if no mapping found
            mapped_tags.append(mapped_tag)

        return mapped_tags
    
    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)
    text = remove_stopwords(text)
    
    def pos_tag_and_lemmatize(text):
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Map POS tags to WordNet tags
        wordnet_tags = map_pos_tags(pos_tags)

        # Lemmatize based on POS tags
        lemmatizer = WordNetLemmatizer()
        lemmas = " ".join([lemmatizer.lemmatize(token, tag) for token, tag in zip(tokens, wordnet_tags)])

        return lemmas
    text = pos_tag_and_lemmatize(text)

    return text


def label_tweet(test_review):
  token_list = tokenizer.texts_to_sequences([test_review])[0]
  token_list = pad_sequences([token_list], maxlen=44, padding='post')
  predicted = model.predict(token_list, verbose=0)
  if predicted >= 0.5:
    return 1
  else: 
    return 0


def analyze_text(comment):
    comment = cleaning(comment)
    result = label_tweet(comment)
    if result == 0: 
        text = "Negative"
    else:
        text = "Positive"
    return text


# It can be used to reconstruct the model identically.
model = keras.models.load_model("twitter_sentiment.keras",
                                custom_objects={'Attention': Attention})

# Load tokenizer
tokenizer = load_tokenizer('tokenizer.pkl')

interface = gr.Interface(fn=analyze_text, inputs=gr.inputs.Textbox(lines=2, placeholder='Enter a positive or negative tweet here...'),
                         outputs='text',title='Twitter Sentimental Analysis', theme='darkhuggingface')
interface.launch(inline=False)