from system_prompt import SYSTEM_PROMPT
from openai import OpenAI
from dotenv import load_dotenv
import os
import math

load_dotenv()

def _extract_content_from_choice(choice):
    if choice is None:
        return None
    msg = getattr(choice, "message", None)
    if msg is None:
        return None
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def openrouter(system_prompt, user_prompt):
    try:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        )

        response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        )

        reply = _extract_content_from_choice(response.choices[0])
        if reply is None:
            raise ValueError("No content in Openrouter response")
        
        return reply, 200
    except Exception as e:
        print("Openrouter Error:", e)
        return str(e), 500


def huggingface(system_prompt, user_prompt):
    try:
        HF_TOKEN = os.getenv("HF_TOKEN")

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
        )

        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
            messages=[
                {"role": "system","content": system_prompt},
                {"role": "user","content": user_prompt}
            ],
        )

        reply = _extract_content_from_choice(completion.choices[0])
        if reply is None:
            raise ValueError("No content in Huggingface response")
        return reply, 200
    except Exception as e:
        print("Huggingface Error:", e)
        return str(e), 500
        

def calculate_clips(duration):
    duration = int(duration)
    clips = math.ceil(duration / 5)
    return max(1, clips)

def seperate_prompts(prompts):
    return prompts.split("\n")


def modify_prompt(user_prompt, duration, audio_prompt=None):
    clips = calculate_clips(duration)
    system_prompt = SYSTEM_PROMPT
    user_prompt_ = "Video Prompt: " + user_prompt
    if audio_prompt:
        user_prompt_+= f"\nAudio Prompt: {audio_prompt}"
        
    user_prompt_ = user_prompt_ + f"\nNumber of Clips: {clips}"
    response, status = openrouter(system_prompt, user_prompt_)
    if status == 200:
        return response
    response, status = huggingface(system_prompt, user_prompt_)
    if status == 200:
        return response
    return seperate_prompts(user_prompt)
