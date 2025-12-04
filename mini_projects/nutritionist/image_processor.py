import base64

def prepare_encoded_image(image_content) :
    # image_content is raw binary data from Gradio File component
    # Encode to base64 first, then wrap with data URI prefix
    if isinstance(image_content, bytes):
        encoded = base64.b64encode(image_content).decode("utf-8")
    else:
        # Already a string (base64 or path)
        encoded = image_content
    
    image_url = f"data:image/jpeg;base64,{encoded}"
    return image_url