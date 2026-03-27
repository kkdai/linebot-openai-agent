import os
import sys

import aiohttp
from fastapi import Request, FastAPI, HTTPException
from openai import AsyncOpenAI
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled
)

from linebot.models import (
    MessageEvent, TextSendMessage
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot import (
    AsyncLineBotApi, WebhookParser
)

# OpenAI Agent configuration
BASE_URL = os.getenv("EXAMPLE_BASE_URL") or ""
API_KEY = os.getenv("EXAMPLE_API_KEY") or ""
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME") or ""

# LINE Bot configuration
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)

# Image processing prompt
image_prompt = '''
Describe this image with scientific detail, reply in zh-TW:
'''

# Validate environment variables
if channel_secret is None:
    print('Specify ChannelSecret as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify ChannelAccessToken as environment variable.')
    sys.exit(1)
if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, "
        "EXAMPLE_MODEL_NAME via env var or code."
    )

# Initialize the FastAPI app for LINEBot
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Initialize OpenAI client
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


@function_tool
def get_weather(city: str):
    """Get weather information for a city"""
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


@function_tool
def translate_to_chinese(text: str):
    """Translate text to Traditional Chinese"""
    print(f"[debug] translating: {text}")
    return f"Translating to Chinese: {text}"


@app.post("/")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue

        if event.message.type == "text":
            # Process text message
            msg = event.message.text
            user_id = event.source.user_id
            print(f"Received message: {msg} from user: {user_id}")

            # Use the user's prompt directly with the agent
            response = await generate_text_with_agent(msg)
            reply_msg = TextSendMessage(text=response)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
        elif event.message.type == "image":
            return 'OK'
        else:
            continue

    return 'OK'


async def generate_text_with_agent(prompt):
    """
    Generate a text completion using OpenAI Agent.
    """
    # Create agent with appropriate instructions
    agent = Agent(
        name="Assistant",
        instructions=(
            "You are a helpful assistant that responds in "
            "Traditional Chinese (zh-TW). "
            "Provide informative and helpful responses."
        ),
        model=OpenAIChatCompletionsModel(
            model=MODEL_NAME, openai_client=client),
        tools=[get_weather, translate_to_chinese],
    )

    try:
        result = await Runner.run(agent, prompt)
        return result.final_output
    except Exception as e:
        print(f"Error with OpenAI Agent: {e}")
        return f"抱歉，處理您的請求時出現錯誤: {str(e)}"
