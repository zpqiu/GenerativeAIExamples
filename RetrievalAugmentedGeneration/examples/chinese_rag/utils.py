import os
import re
import logging
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter

logger = logging.getLogger(__name__)

RANKING_MODEL = os.environ.get("RANKING_MODEL", "BAAI/bge-reranker-v2-m3")
NIM_MAX_LENGTH = 900

@lru_cache
def get_ranking_model():
    """Create the ranking model."""
    logger.info(f"Using {RANKING_MODEL} as model engine for ranking")
    tokenizer_rerank = AutoTokenizer.from_pretrained(RANKING_MODEL)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(RANKING_MODEL)
    if torch.cuda.is_available():
        rerank_model = rerank_model.to('cuda')
    rerank_model.eval()
    return rerank_model, tokenizer_rerank


def get_text_splitter(embedding_model_name="BAAI/bge-large-zh-v1.5", chunk_size=510, chunk_overlap=200):
    """Get the text splitter function."""
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter

prompt_template = "请你对从 PDF 中解析的文本进行 Markdown 格式化。下面的文本是使用 PDF 解析工具解析出的文本：\n\n```{pdf_text}```\n\n请将其转换为 Markdown 格式, 格式化的结果用```markdown ```包裹。"


def _format_by_llm(client, chunk_text):
    prompt = prompt_template.format(pdf_text=chunk_text)
    completion = client.chat.completions.create(
        model="meta/llama3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )

    formatted_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            formatted_text += chunk.choices[0].delta.content
    
    # use re extract the formatted text (wrapped by ```markdown```) from the completion response
    formatted_text = re.search(r"```markdown(.*?)```", formatted_text, re.DOTALL).group(1)
    return formatted_text


def markdown_pdf_document(client, document):
    """Convert the PDF document to markdown."""
    # Split the document into chunks by token number = NIM_MAX_LENGTH
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama3-ChatQA-1.5-8B")
    tokens = tokenizer(document.page_content, add_special_tokens=False)["input_ids"]
    number_of_failed_chunks = 0
    formatted_chunks = []

    for i in range(0,  len(tokens), NIM_MAX_LENGTH):
        chunk = tokenizer.decode(tokens[i:i + NIM_MAX_LENGTH])
        try:
            formatted_chunk = _format_by_llm(client, chunk)
            if formatted_chunk and len(formatted_chunk) > 0:
                formatted_chunks.append(formatted_chunk)
            else:
                formatted_chunks.append(chunk)
        except Exception as e:
            logger.error(f"Failed to format chunk: {e}")
            formatted_chunks.append(chunk)
            number_of_failed_chunks += 1
    
    document.page_content = " ".join(formatted_chunks)
    return document