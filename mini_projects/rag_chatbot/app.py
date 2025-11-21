from document_manager import load_and_store_document
from vector_store_manager import VectorStoreManager
from chatbot import Chatbot
from helper import generate_hash
import gradio as gr


doc_url = "https://www.livescience.com/11375-top-ten-conspiracy-theories.html"
collection_name = generate_hash(doc_url.strip())

load_and_store_document(url=doc_url)
vector_store_manager = VectorStoreManager(collection_name)
chatbot = Chatbot(vector_store_manager.get_retriever())

app = gr.Interface(fn=chatbot.chat, inputs=[gr.Textbox()], outputs=[gr.TextArea()])

app.launch()
