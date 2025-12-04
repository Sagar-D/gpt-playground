import gradio as gr
from gradio.components import Textbox, File
from llm_manager import nutritionist_chain
from image_processor import prepare_encoded_image

def consult_nutrition_bot(image_data):
    image_url = prepare_encoded_image(image_data)
    response = nutritionist_chain.invoke({"image_url":image_url})
    return response


app = gr.Interface(
    fn=consult_nutrition_bot,
    inputs=[
        # Textbox(placeholder="Input Prompt"),
        File(file_count="single", file_types=["png", "jpeg"], type="binary"),
    ],
    outputs=Textbox(placeholder="Model Response :", lines=20),
)

if __name__ == "__main__":
    app.launch()
