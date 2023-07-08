import keras
import pickle
import tempfile
import numpy as np
import gradio as gr
import tensorflow as tf
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


def label_tweet(test_review):
  test_review = test_review.lower().strip()
  token_list = tokenizer.texts_to_sequences([test_review])[0]
  token_list = pad_sequences([token_list], maxlen=44, padding='post')
  predicted = model.predict(token_list, verbose=0)
  if predicted >= 0.5:
    return 1
  else: 
    return 0


def analyze_text(comment):
    
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