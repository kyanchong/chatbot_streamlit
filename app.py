import streamlit as st
#from streamlit_chat import message as st_message
from streamlit_chat import message as st_message
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

st.title("Chatbot Blenderbot Streamlit")

if "history" not in st.session_state:
    st.session_state.history = []
    
def get_models():
  tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
  model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
  return tokenizer, model
  
def generate_answer():
    tokenizer, model = get_models()
    user_message = st.session_state.input_text
    inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
    result = model.generate(**inputs)
    message_bot = tokenizer.decode(result[0], skip_special_tokens=True)  # .replace("<s>", "").replace("</s>", "")
    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})
    
st.text_input("Response", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)