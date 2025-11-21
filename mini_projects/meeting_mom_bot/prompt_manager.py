from langchain_core.prompts import PromptTemplate

MEETING_MOM_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant whose job is to understand the meeting transcripts and create meeting summury and Minutes of Meeting (MoM).
Minutes of Meeting should include :
- Agenda of the meeting
- Short meeting Summary
- Key decisions made
- Action Items (and owner of the action item, if mentioned. Ignore otherwise)
- Open Items for future discuss

Important Notes :
1. Do not hallucinate! Create the MoM only based on the transcript provided
2. Keep the MoM precise and pointer wise.
3. Don't send any other additional text in the response other than MoM
4. If there isn't enought data to populate any section of MoM, skip that section.
5. Based on semantic understading, populate pointers for sections like Key decisions made, Action Items etc

Meeting Transcript : 
{meeting_transcript}

Response : 

** Meeting MoM **
"""
)

