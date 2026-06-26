from openai import OpenAI
from pprint import pprint
import os
def get_weather():
    return "666.6C"

def main():
    client = OpenAI(
        base_url="https://api.deepinfra.com/v1/",
        api_key=os.environ.get("API_KEY"),

    )


    resp = client.chat.completions.create(
        model="Qwen/Qwen3.6-35B-A3B",
        messages=[
            {"role": "system", "content": "Türkçe konuş. Haber spikeri gibi konuş. Çok kısa konuş."},
            {"role": "user", "content": "Selamlar! Istanbul'da havalar nasıl?"},
            {"role": "assistant", "content": ""}, # tool call verdi
            {"role": "tool", "tool_call_id":"chatcmpl-tool-a1a689929afbbb5d", "content": get_weather()}
        ],
        temperature=0.0,
        top_p=0.95,
        max_tokens=12000,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a loc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
        ],
        tool_choice="auto"
    )


    pprint(resp.model_dump())
    print("\n======")

    pprint(resp.choices[0].message.content)




if __name__ == "__main__":
    main()
