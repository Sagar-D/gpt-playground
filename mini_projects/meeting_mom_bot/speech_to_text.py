from transformers import pipeline

def speech_to_text(audio_file_path:str) -> str :

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30
    )

    print("Interpreting audio file using transformer")
    prediction = pipe(audio_file_path, batch_size=8)
    print("Interpretation completed successfully!!")
    
    return prediction["text"]


if __name__ == "__main__" :

    file_path = "mini_projects/meeting_mom_bot/data/sample_meeting_rec_1_min.wav"
    print(speech_to_text(file_path))
