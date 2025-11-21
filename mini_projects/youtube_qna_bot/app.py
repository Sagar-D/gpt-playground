from youtube_helper import fetch_transcript, extract_video_id
from retriever import create_vector_index, retrieve_docs
from pprint import pprint
from llm import chat_chain

video_url = "https://www.youtube.com/watch?v=akvLbLfKSIo"
video_id = extract_video_id(video_url)
transcript = fetch_transcript(video_id)

vector_index = create_vector_index(transcript, video_id)
response = chat_chain.invoke({"vector_index":vector_index,"prompt":"Is Yonex BG80 badminton string good for power hitting"})
pprint(response)

