from openai import OpenAI
from pprint import pprint
import os

from openai.types.chat import ChatCompletionMessageParam
def get_weather():
    return "-300K"

def main():
    client = OpenAI(
        base_url="https://api.deepinfra.com/v1/",
        api_key=os.environ.get("API_KEY"),

    )

    message_history: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "Gıbrıslı gardaşçıksin, bütün gün eşşeğin üstünde gezer. Gıbrıs şivesi ile konuşun. Annadıng?"}
    ]

    quit_loop = False

    while not quit_loop:

        user_input: str = input("User:> ")
        message_history.append({"role": "user", "content": user_input})

        resp = client.chat.completions.create(
            model="Qwen/Qwen3.6-35B-A3B",
            messages=message_history,
            temperature=0.8,
            top_p=0.95,
            max_tokens=12000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        assistant_message_str = (resp.choices[0].message.content or "(No Message)").strip()

        message_history.append({
            "role": "assistant",
            "content": assistant_message_str
        })
        
        print("Assistant:> " + assistant_message_str)


    




if __name__ == "__main__":
    main()
