import argparse
from dataclasses import dataclass
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import utils
import time 
from dotenv import load_dotenv

load_dotenv()

API_KEY = ********************
os.environ["AZURE_OPENAI_API_KEY"] = API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = ********************
os.environ["AZUREOPENAI_API_TYPE"] = ********************
os.environ["OPENAI_API_VERSION"] = ********************
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDING_MODEL")

PROMPT_TEMPLATE = """
You are working on classifying the accuracy of engagements between an agent and the responses sent to an end-user given a document, formatted as `<|Prompt|>` and `<|Response|>`. 

Accuracy (1-10)
1-2: The response is mostly or entirely incorrect.
3-4: The response has significant inaccuracies but also some correct elements.
5-6: The response is mostly correct but contains minor errors or omissions.
7-8: The response is correct with very few errors or omissions.
9-10: The response is entirely correct, with no errors or omissions.
Completeness (1-10)
1-2: The response is missing most of the expected information.
3-4: The response includes some expected information but is largely incomplete.
5-6: The response includes most of the expected information but may miss some key elements.
7-8: The response includes all expected information but lacks depth or detail.
9-10: The response is thorough and includes all expected information in detail.
Adherence (1-10)
1-2: The response includes significant amounts of information from outside the source material.
3-4: The response includes some information from outside the source material.
5-6: The response is mostly based on the source material but includes minor irrelevant content.
7-8: The response is based on the source material with very little irrelevant content.
9-10: The response strictly adheres to the source material with no irrelevant content.
Clarity (1-10)
1-2: The response is confusing and difficult to understand.
3-4: The response is somewhat understandable but contains significant readability issues.
5-6: The response is mostly clear but could be improved in terms of readability and understandability.
7-8: The response is clear and understandable with minor readability issues.
9-10: The response is exceptionally clear, well-structured, and easy to understand.
Level of Detail (1-10)
1-2: The response lacks detail and is overly brief.
3-4: The response includes some detail but is still insufficient.
5-6: The response includes a moderate level of detail but could be more comprehensive.
7-8: The response includes a good level of detail but may include some unnecessary information.
9-10: The response is detailed and comprehensive without including unnecessary information.
Overall Score (1-10): The overall score will be the averages of all the calcaulted scores. Overall Score = ((Accuracy+Completeness+Adherence+Clarity+Level of Detail)/5)
1-2: The response is generally of poor quality and not useful.
3-4: The response has some value but is largely inadequate.
5-6: The response is somewhat useful but needs significant improvement.
7-8: The response is useful and acceptable with minor improvements needed.
9-10: The response is of high quality and could not be significantly improved.
Based on the following context, rate the accuracy of the reponse (0 for inaccurate, 1 for accurate), and rate the quality of the response from 1-10 based on the overall score calculated from the above rubric

If the response is "Unable to find matching results.", it is automatically inaccurate and all quality scores should be 0.

{context}

---

Provide two values: First value: Accuracy (Return 1 for "ACCURATE", Return 0 for "INACCURATE) and Second value: an estimate of the quality of the response given as the overall score of the scoring rubric. 
Format your response such that you give only give the acurracy and quality estimate, separated by a comma. No additional text/explanation is needed.  {question}

"""

def get_scores(db, input_text, size, overlap):
    CHROMA_PATH = utils.get_chroma_folder_name(EMBEDDINGS_MODEL_NAME, size, overlap)
    openai_api_key = ********************

    # # Prepare the DB.
    # embedding_function = AzureOpenAIEmbeddings(
    #     openai_api_key=openai_api_key,
    #     model=EMBEDDINGS_MODEL_NAME)   
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(input_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=input_text)

    model = AzureChatOpenAI(
        model="gpt-4",
        azure_deployment="fs-gpt-4"
    )
    
    response_text = model.predict(prompt)
    return response_text