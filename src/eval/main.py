"""Retrieval and answer evaluation entry point.

Loads a small eval set of ``{user_input, reference}`` pairs, runs the same
:class:`Retriever` used at inference time to fetch ``retrieved_contexts`` for
each question, generates an answer from those contexts, then scores quality
with three ragas metrics:

* ``ContextRecall``      — did retrieval surface the reference information?
* ``ContextPrecision``   — are the retrieved chunks relevant (low noise)?
* ``AnswerCorrectness``  — is the generated answer factually/semantically right?

Run it as a module::

    python -m src.eval.main --dataset eval.json

Note: ``AnswerCorrectness`` requires a generated answer, so a ``--chat-model``
is used to produce one per question before scoring.
"""

import json

import click
from openai import AsyncOpenAI, OpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)

from ..ingestion.embed import OpenAIEmbedder
from ..inference.cli import _SYSTEM_PROMPT, _format_context
from ..inference.retrieval import RetrievedChunk, Retriever

# model="zai-org/GLM-5.2"
# model="Qwen3.6-35B-A3B-NVFP4"


def _load_dataset(path: str) -> list[dict[str, str]]:
    """Read eval samples: a JSON list of ``{"user_input", "reference"}``."""
    with open(path) as f:
        samples = json.load(f)
    for s in samples:
        if "user_input" not in s or "reference" not in s:
            raise ValueError(
                "each sample needs 'user_input' and 'reference' keys, got "
                f"{sorted(s)}"
            )
    return samples


def _generate_answer(
    client: OpenAI,
    chat_model: str,
    question: str,
    chunks: list[RetrievedChunk],
    temperature: float,
) -> str:
    """Answer ``question`` grounded in ``chunks`` using the chat model.

    Mirrors the single-turn prompting used by the inference CLI so the answer
    being scored reflects what the deployed RAG pipeline would produce.
    """
    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


@click.command()
@click.option(
    "--dsn",
    envvar="DATABASE_DSN",
    required=True,
    help="libpq DSN, e.g. postgresql://user:pass@host:5432/db",
)
@click.option(
    "--api-key",
    envvar="API_KEY",
    required=True,
    help="API key for the embeddings and chat endpoints.",
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    show_default=True,
    help="Base URL of the OpenAI-compatible API.",
)
@click.option(
    "--embedding-model",
    envvar="EMBEDDING_MODEL",
    required=True,
    help="Embedding model id (must match the model used at ingestion).",
)
@click.option(
    "--embedding-size",
    envvar="EMBEDDING_SIZE",
    type=int,
    required=True,
    help="Embedding dimensionality (must match the schema's VECTOR size).",
)
@click.option(
    "--eval-model",
    envvar="EVAL_MODEL",
    default="Qwen3.6-35B-A3B-NVFP4",
    show_default=True,
    help="Chat model id used by the ragas judge.",
)
@click.option(
    "--chat-model",
    envvar="CHAT_MODEL",
    required=True,
    help="Chat model id used to generate answers scored by answer correctness.",
)
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file: a list of {user_input, reference} objects.",
)
@click.option(
    "--top-k",
    envvar="TOP_K",
    default=15,
    show_default=True,
    type=int,
    help="Number of chunks to retrieve per question.",
)
@click.option(
    "--temperature",
    envvar="TEMPERATURE",
    default=0.2,
    show_default=True,
    type=float,
    help="Sampling temperature for answer generation.",
)
def main(
    dsn: str,
    api_key: str,
    base_url: str,
    embedding_model: str,
    embedding_size: int,
    eval_model: str,
    chat_model: str,
    dataset_path: str,
    top_k: int,
    temperature: float,
) -> None:
    """Score retrieval and answer quality over a dataset."""
    dataset = _load_dataset(dataset_path)

    aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
    client = OpenAI(api_key=api_key, base_url=base_url)
    llm = llm_factory(model=eval_model, client=aclient, max_tokens=10000000, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    eval_embeddings = embedding_factory(
        "openai",
        model=embedding_model,
        client=aclient,
        interface="modern",
    )

    context_recall = ContextRecall(llm)
    context_precision = ContextPrecision(llm)
    answer_correctness = AnswerCorrectness(llm=llm, embeddings=eval_embeddings)

    embedder = OpenAIEmbedder(
        client=client,
        model=embedding_model,
        dimensions=embedding_size,
    )

    with Retriever.connect(dsn, embedder) as retriever:
        # Retrieve contexts and generate an answer for each sample, then
        # batch-score every metric across the whole dataset.
        retrieval_inputs: list[dict[str, object]] = []
        answer_inputs: list[dict[str, object]] = []
        for sample in dataset:
            chunks: list[RetrievedChunk] = retriever.search(
                sample["user_input"], top_k=top_k
            )
            answer = _generate_answer(
                client, chat_model, sample["user_input"], chunks, temperature
            )
            retrieval_inputs.append(
                {
                    "user_input": sample["user_input"],
                    "reference": sample["reference"],
                    "retrieved_contexts": [c.content for c in chunks],
                }
            )
            answer_inputs.append(
                {
                    "user_input": sample["user_input"],
                    "response": answer,
                    "reference": sample["reference"],
                }
            )

        recall_results = context_recall.batch_score(retrieval_inputs)
        print(recall_results)
        precision_results = context_precision.batch_score(retrieval_inputs)
        print(precision_results)
        correctness_results = answer_correctness.batch_score(answer_inputs)
        print(correctness_results)

    recalls: list[float] = []
    precisions: list[float] = []
    correctnesses: list[float] = []
    click.echo(f"{'recall':>7} {'prec':>7} {'correct':>7}  question")
    for sample, rec, prec, corr in zip(
        dataset, recall_results, precision_results, correctness_results
    ):
        r, p, c = float(rec.value), float(prec.value), float(corr.value)
        recalls.append(r)
        precisions.append(p)
        correctnesses.append(c)
        click.echo(f"{r:7.3f} {p:7.3f} {c:7.3f}  {sample['user_input']}")

    if dataset:
        n = len(dataset)
        click.echo(
            f"\nmean context_recall:     {sum(recalls) / n:.3f}"
            f"\nmean context_precision:  {sum(precisions) / n:.3f}"
            f"\nmean answer_correctness: {sum(correctnesses) / n:.3f}"
        )


if __name__ == "__main__":
    main()
