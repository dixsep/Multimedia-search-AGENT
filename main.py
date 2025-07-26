from langgraph.graph import StateGraph, START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import json
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from basic import GEMINI_API_KEY, GEMINI_API_KEY1
import base64
import google.genai as genai
from dotenv import load_dotenv
from google.genai import types

'''
AIM : user asks to detect object in a file
AGENT : should search file and return whether there is an object in the file
'''

model = "gemini-2.5-pro"

client = genai.Client(api_key = GEMINI_API_KEY1)

llm = ChatGoogleGenerativeAI(
    model = model,
    google_api_key = GEMINI_API_KEY
)

class Classify(BaseModel):

    audio_score : float = Field(description = "confidence score of audio search between 0 and 1")
    video_score : float = Field(description = "confidence score of video search between 0 and 1")
    image_score : float = Field(description = "confidence score of image search between 0 and 1")


class State(TypedDict):
    messages : Annotated[list, add_messages]
    next : None | str
    message_type : None | str



#======================IMAGE TOOL==========
@tool
def load_image(imagePath : str, question : str):
    '''
    Detect object in the Image file
    :param imagePath:  path to the image file
    :param question: question that the user asks about object
    :return:
    '''

    with open(imagePath, "rb") as image_file:
        image_bytes = image_file.read()


    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            types.Part(text = f"{question}explain your reasoning")
        ]
    )

    return response.text


#======================VIDEO TOOL=========
@tool
def load_video(videoPath : str, question : str):

    '''
    Detect object in the video file
    :param videoPath: path to the video file
    :param question: question that the user asks about file
    :return:
    '''

    video_bytes = open(videoPath, "rb").read()

    response = client.models.generate_content(
        model = 'gemini-2.5-flash',
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text= f"{question} explain your reasoning")
            ]
        )
    )

    return response.text

#======================AUDIO TOOL=========
@tool
def load_audio(audioPath : str, question : str):

    '''
    Detect object in the audio file
    :param audioPath: path to the audio file
    :param question: question that the user asks about audio file
    :return:
    '''

    with open(audioPath, "rb") as audio_file:
        audio_bytes = audio_file.read()

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part(text = f"{question} explain your reasoning"),
            types.Part.from_bytes(
                data = audio_bytes,
                mime_type = 'audio/mp3',
            )
        ]
    )

    return response.text


tools = [load_image, load_video, load_audio]
llm_with_tools = llm.bind_tools(tools)

#======================CLASSIFIER AGENT=========
def classifier(state : State):

    system_msg = SystemMessage(content = "You are a classifier agent, ypu should determine whether user is asking to search in video file or audio file or image file")
    all_msg = [system_msg] + state["messages"]

    response = llm.with_structured_output(Classify).invoke(all_msg)  # no need to use tools

    print(response)

    if response.audio_score >= response.video_score and response.audio_score >= response.image_score:
        return {"message_type" : "audio"}
    elif response.audio_score <= response.video_score and response.image_score <= response.video_score:
        return {"message_type" : "video"}
    else:
        return {"message_type" : "image"}

#======================ROUTER AGENT=========
def router(state : State):

    message_type = state.get("message_type", "image")

    if message_type == "audio":
        return {"next" : "audio"}
    elif message_type == "video":
        return {"next" : "video"}
    else:
        return {"next" : "image"}

#======================IMAGE AGENT=========
def image_agent(state : State):

    system_msg = SystemMessage(content = "You are a image agent, you are a expert at detecting objects in images, Only provide answer to image file"
                                         "by using image tools")
    all_msg = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(all_msg)
    print(response.content)

    return {"messages" : [response]}

#======================VIDEO AGENT=========
def video_agent(state : State):

    system_msg = SystemMessage(content = "You are a video agent, you are a expert at detecting objects in videos, Only provide answer to video file"
                                         "by using video tools")
    all_msg = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(all_msg)
    print(response.content)

    return {"messages" : [response]}

#======================AUDIO AGENT=========
def audio_agent(state : State):

    system_msg = SystemMessage(content = "You are a audio agent, you are a expert at detecting objects in audios, Only provide answer to audio file"
                                         "by using audio tools")
    all_msg = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(all_msg)
    print(response.content)

    return {"messages" : [response]}


#================GRAPH=======================

'''
  start -> classifier -> router -> (image / audio / video) + tools -> end
  
  tools can be seperate or as a whole 
  
  check ??
  
  working with whole ?
'''

graph_builder = StateGraph(State)

#====================NODES================
graph_builder.add_node("classifier", classifier)
graph_builder.add_node("router", router)
graph_builder.add_node("image_agent", image_agent)
graph_builder.add_node("video_agent", video_agent)
graph_builder.add_node("audio_agent", audio_agent)
graph_builder.add_node("tools", ToolNode(tools))

#====================EDGES================
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state : state.get("next"),
    {"audio" : "audio_agent", "video" : "video_agent", "image" : "image_agent"}
)
graph_builder.add_conditional_edges("image_agent", tools_condition)
graph_builder.add_edge("tools", "image_agent")
graph_builder.add_conditional_edges("video_agent", tools_condition)
graph_builder.add_edge("tools", "video_agent")
graph_builder.add_conditional_edges("audio_agent", tools_condition)
graph_builder.add_edge("tools", "audio_agent")
graph_builder.add_edge("image_agent", END)
graph_builder.add_edge("video_agent", END)
graph_builder.add_edge("audio_agent", END)

##====================COMPILE================
graph = graph_builder.compile()


user_input = "Can you hear cat in dog_audio_yes.mp3 file"
state = {"messages" : [HumanMessage(content = user_input)]}

state = graph.invoke(state)
print(state["messages"][-1].content)