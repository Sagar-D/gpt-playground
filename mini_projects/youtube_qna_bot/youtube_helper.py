from youtube_transcript_api import YouTubeTranscriptApi
from pprint import pprint
import re

def extract_video_id(video_url) :

    video_url=video_url.strip()

    if video_url.startswith("http") :
        pattern_match = re.search(r"v=[A-Za-z0-9]+", video_url)
        return pattern_match.group()[2:]
    elif video_url.isalnum():
        return video_url
    else :
        raise Exception("Invalid youtube video url passed!!")


def fetch_transcript(video_id:str) :
    
    transcript_loader = YouTubeTranscriptApi()
    transcript_list = transcript_loader.fetch(video_id=video_id)

    return _create_transcript_string(transcript_list.snippets)

def _create_transcript_string(transcripts:list) :

    transcript_string = ""
    for transcript in transcripts :
        transcript_string += transcript.text + " "
    return transcript_string

if __name__ == '__main__' :
    video_url = "https://www.youtube.com/watch?v=vSQjk9jKarg"
    pprint(fetch_transcript(video_url))