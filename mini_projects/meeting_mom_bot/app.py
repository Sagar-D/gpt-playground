import gradio as gr
from gradio import Audio, TextArea
from llm_manager import mom_chain


def minutes_if_meeting(audio_file_path:str) -> str :
    return mom_chain.invoke(input=audio_file_path)

app = gr.Interface(
    minutes_if_meeting,
    inputs=Audio(label="Upload meeting recording in wav or mp3 format", type="filepath"),
    outputs=TextArea(placeholder="Minutes of Meeting : ")
)

if __name__ == "__main__" :
    app.launch()

