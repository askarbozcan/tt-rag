import os
from textwrap import dedent


import numpy as np
from openai import OpenAI
from pprint import pprint
from tqdm import tqdm

from openai.types.chat import ChatCompletionMessageParam

corpus = [
    """Mustafa Kemal Atatürk[d] (1881,[e] Selanik - 10 Kasım 1938, İstanbul), Türk mareşal, devlet adamı, yazar, Türk Kurtuluş Savaşı'nın başkomutanı, Türkiye Cumhuriyeti'nin kurucusu ve ilk cumhurbaşkanıdır. Türkiye'yi laik, sanayileşen bir ulusa dönüştüren kapsamlı ilerici reformlar üstlenmiştir.[6] İdeolojik olarak sekülarist ve milliyetçi politikaları ve sosyo-politik teorileri Kemalizm olarak tanınmıştır.[6]""",
    """İstanbul, Türkiye'nin başkenti, Türkiye'nin ekonomik, kültürel ve tarihî merkezini oluşturan en kalabalık şehridir. 15,7 milyonu aşan nüfusuyla Türkiye nüfusunun yaklaşık %18,3'ine ev sahipliği yapmaktadır.[4] İstanbul, Avrupa'daki en kalabalık şehirlerden biri olmasının yanı sıra, dünya genelinde de nüfus bakımından en kalabalık şehirler arasında yer alır. İstanbul, iki kıtada yer alan bir şehir olup, nüfusunun yaklaşık üçte ikisi Avrupa yakasında, geri kalanı ise Asya yakasında yaşamaktadır.[8] Şehir; Türkiye'nin kuzeybatısında, Marmara Denizi ile Karadeniz arasında yer alan ve dünyanın en işlek su yollarından biri olan Boğaziçi boyunca uzanır. 5.461 km² yüzölçümüne sahip olan İstanbul'un idari sınırları, İstanbul ili ile örtüşmektedir.[9]""",
    """Ankara, Türkiye'nin yönetildiği şehir, Ankara ilinin merkezi olan şehirdir.[3] Coğrafi olarak Türkiye'nin merkezine yakın bir konumda bulunur ve İç Anadolu Bölgesi'nde yer alır. 5,5 milyona yaklaşan nüfusuyla Ülkenin İstanbul'dan sonra nüfus bakımından ikinci büyük şehridir."""
]

embeddings = [

]



def embed(client: OpenAI, strs: list[str]) -> list[np.ndarray]:
    emb_resp = client.embeddings.create(
        input=strs,
        model="BAAI/bge-m3"
    )

    result_embeddings = []
    for emb_obj in emb_resp.data:
        result_embeddings.append(
            np.array(emb_obj.embedding, dtype=np.float32)
        )
    
    return result_embeddings


def main():
    client = OpenAI(
        base_url="https://api.deepinfra.com/v1/",
        api_key=os.environ.get("API_KEY"),

    )

    corpus_embeddings = embed(client, corpus)


    prompt = dedent("""
    You are a helpful assistant.
    You only reply using the information from the sources
    provided to you. Sources will be given to you.
    DO NOT WRITE ANYTHING BASED ON YOUR OWN KNOWLEDGE. Only use provided
    texts, even if you think they are wrong.

    Reply in Turkish.
    """)

    message_history: list[ChatCompletionMessageParam] = [
        {"role": "system", 
        "content": prompt}
    ]


    quit_loop = False
    is_first_loop = True
    while not quit_loop:
        user_input: str = input("User:> ")

        # rag
        if is_first_loop:
            query_emb = embed(client, [user_input])[0]

            scores = []
            for i,emb in enumerate(corpus_embeddings):
                scores.append(
                    np.dot(emb, query_emb) / (np.linalg.norm(emb) * np.linalg.norm(query_emb))
                )

            max_i = -1
            max_score = 0
            for i,score in enumerate(scores):
                if score > max_score:
                    max_i = i
                    max_score = score
            

            print("RELEVANT CHUNK INDEX IS", max_i)
            relevant_chunk = corpus[max_i]

            prompt= f"""SOURCES: Most relevant documents: {"\n".join(corpus)}\n User query: {user_input}"""
            message_history.append({"role": "user", "content": prompt})
        else:
            message_history.append({"role": "user", "content": user_input})
        




        resp = client.chat.completions.create(
            model="Qwen/Qwen3.6-35B-A3B",
            messages=message_history,
            temperature=0.1,
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
