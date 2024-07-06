# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Generator, List, Dict, Any

import re
import shutil
import requests
import torch
from openai import OpenAI
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from RetrievalAugmentedGeneration.common.base import BaseExample
from RetrievalAugmentedGeneration.common.utils import get_config, get_llm, get_embedding_model, create_vectorstore_langchain, get_docs_vectorstore_langchain, del_docs_vectorstore_langchain, get_vectorstore
from RetrievalAugmentedGeneration.common.tracing import langchain_instrumentation_class_wrapper
from RetrievalAugmentedGeneration.example.utils import get_ranking_model, markdown_pdf_document, get_text_splitter

logger = logging.getLogger(__name__)
# DOCS_DIR = os.path.abspath("./uploaded_files")
vector_store_path = "vectorstore.pkl"
document_embedder = get_embedding_model()
document_reranker, tokenizer_rerank = get_ranking_model()
text_splitter = None
settings = get_config()

# mainly for tracking the status of uploaded but still not completely processed files
uploaded_files_status = {}

nim_client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ.get("NVIDIA_API_KEY")
)

RECALL_K = 10

try:
    vectorstore = create_vectorstore_langchain(document_embedder=document_embedder)
except Exception as e:
    vectorstore = None
    logger.info(f"Unable to connect to vector store during initialization: {e}")


def rank_docs(docs, question, max_return=3):
    if not docs or len(docs) == 0:
        return []
    rerank_pairs = [[question, doc.page_content] for doc in docs]
    inputs = tokenizer_rerank(rerank_pairs, padding=True, truncation=True, 
                              return_tensors='pt', max_length=1024).to(document_reranker.device)

    with torch.no_grad():
        outputs = document_reranker(**inputs.to(document_reranker.device))
    logits = outputs.logits.view(-1, ).float()
    rank = logits.argsort(descending=True)
    selected_docs = [docs[i] for i in rank[:max_return]]
    return selected_docs


def ingest_pdf(pdf_path):
    # copy pdf file to /pdfs
    pdf_filename = os.path.basename(pdf_path)
    paring_pdf_path = f"/pdfs/{pdf_filename}"
    output_path = f"/pdfs/{pdf_filename[:-4]}"
    # os.mkdir(output_path, exist_ok=True)
    os.makedirs(output_path, mode=0o777, exist_ok=True)
    shutil.copy(pdf_path, paring_pdf_path)
    req = requests.post("http://pdf-parser:8088/parse_pdf", 
                    json={"pdf_path": paring_pdf_path, "output_path": output_path})

    if req.status_code == 200:
        logger.info(f"PDF parsing successful: {pdf_filename}")
        md_path = f"{output_path}/doc.md"
        target_md_path = f"{pdf_path}.md"
        shutil.copy(md_path, target_md_path)
        raw_documents = TextLoader(target_md_path).load()
        return raw_documents
    else:
        logger.error(f"Failed to parse PDF: {pdf_filename}")
        raw_documents = UnstructuredFileLoader(pdf_path).load()
        return raw_documents


@langchain_instrumentation_class_wrapper
class NvidiaAPICatalog(BaseExample):
    def ingest_docs(self, filepath: str, filename: str):
        """Ingest documents to the VectorDB."""
        global uploaded_files_status
        tracking_filename = os.path.splitext(os.path.basename(filepath))[0]
        if  not filename.endswith((".txt",".pdf",".md",".html")):
            raise ValueError(f"{filename} is not a valid Text, PDF or Markdown file")
        try:
            # Load raw documents from the directory
            # Data is copied to `DOCS_DIR` in common.server:upload_document
            _path = filepath
            uploaded_files_status[tracking_filename] = 0
            if filename.endswith(".md"):
                raw_documents = TextLoader(_path).load()
            elif filename.endswith(".pdf"):
                raw_documents = ingest_pdf(_path)
            else:
                raw_documents = UnstructuredFileLoader(_path).load()

            if raw_documents:
                # if filename.endswith(".pdf"):
                #     raw_documents = [markdown_pdf_document(nim_client, doc) for doc in raw_documents]

                global text_splitter
                if not text_splitter:
                    text_splitter = get_text_splitter(
                        embedding_model_name=settings.embeddings.model_name, 
                        chunk_size=settings.text_splitter.chunk_size - 2, 
                        chunk_overlap=settings.text_splitter.chunk_overlap)

                documents = text_splitter.split_documents(raw_documents)
                vs = get_vectorstore(vectorstore, document_embedder)
                vs.add_documents(documents)
                uploaded_files_status[tracking_filename] = 1
            else:
                uploaded_files_status[tracking_filename] = -1
                logger.warning("No documents available to process!")
        except Exception as e:
            uploaded_files_status[tracking_filename] = -1
            logger.error(f"Failed to ingest document due to exception {e}")
            raise ValueError("Failed to upload document. Please upload an unstructured text document.")

    def llm_chain(
        self, query: str, chat_history: List["Message"], **kwargs
    ) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""

        logger.info("Using llm to generate response directly without knowledge base.")
        # WAR: Disable chat history (UI consistency).
        chat_history = []
        system_message = [("system", settings.prompts.chat_template)]
        conversation_history = [(msg.role, msg.content) for msg in chat_history]
        user_input = [("user", "{input}")]

        # Checking if conversation_history is not None and not empty
        prompt_template = ChatPromptTemplate.from_messages(
            system_message + conversation_history + user_input
        ) if conversation_history else ChatPromptTemplate.from_messages(
            system_message + user_input
        )

        llm = get_llm(**kwargs)

        chain = prompt_template | llm | StrOutputParser()
        augmented_user_input = (
            "\n\nQuestion: " + query + "\n"
        )
        return chain.stream({"input": augmented_user_input}, config={"callbacks":[self.cb_handler]})

    def rag_chain(self, query: str, chat_history: List["Message"], **kwargs) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above."""

        logger.info("Using rag to generate response from document")
        # WAR: Disable chat history (UI consistency).
        chat_history = []
        system_message = [("system", settings.prompts.rag_template)]
        conversation_history = [(msg.role, msg.content) for msg in chat_history]
        user_input = [("user", "{input}")]

        # Checking if conversation_history is not None and not empty
        prompt_template = ChatPromptTemplate.from_messages(
            system_message + conversation_history + user_input
        ) if conversation_history else ChatPromptTemplate.from_messages(
            system_message + user_input
        )

        llm = get_llm(**kwargs)

        chain = prompt_template | llm | StrOutputParser()

        try:
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs != None:
                # try:
                #     logger.info(f"Getting retrieved top k values: {settings.retriever.top_k} with confidence threshold: {settings.retriever.score_threshold}")
                #     retriever = vs.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": settings.retriever.score_threshold, "k": RECALL_K})
                #     docs = retriever.get_relevant_documents(query, callbacks=[self.cb_handler])
                # except NotImplementedError:
                #     # Some retriever like milvus don't have similarity score threshold implemented
                #     retriever = vs.as_retriever()
                #     docs = retriever.get_relevant_documents(query, callbacks=[self.cb_handler])

                retriever = vs.as_retriever(search_kwargs={"k": RECALL_K})
                docs = retriever.get_relevant_documents(query, callbacks=[self.cb_handler])

                if not docs:
                    logger.warning("Retrieval failed to get any relevant context")
                    return iter(["No response generated from LLM, make sure your query is relavent to the ingested document."])

                # reranking
                docs = rank_docs(docs, query)
                docs = docs[:settings.retriever.top_k]

                context = ""
                for doc in docs:
                    context += doc.page_content + "\n\n"

                augmented_user_input = (
                    "Context: " + context + "\n\nQuestion: " + query + "\n"
                )

                return chain.stream({"input": augmented_user_input}, config={"callbacks":[self.cb_handler]})
        except Exception as e:
            logger.warning(f"Failed to generate response due to exception {e}")
        logger.warning(
            "No response generated from LLM, make sure you've ingested document."
        )
        return iter(
            [
                "No response generated from LLM, make sure you have ingested document from the Knowledge Base Tab."
            ]
        )

    def document_search(self, content: str, num_docs: int) -> List[Dict[str, Any]]:
        """Search for the most relevant documents for the given search parameters."""

        try:
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs != None:
                # try:
                #     retriever = vs.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": settings.retriever.score_threshold, "k": settings.retriever.top_k})
                #     docs = retriever.get_relevant_documents(content, callbacks=[self.cb_handler])
                # except NotImplementedError:
                #     # Some retriever like milvus don't have similarity score threshold implemented
                #     retriever = vs.as_retriever()
                #     docs = retriever.get_relevant_documents(content, callbacks=[self.cb_handler])

                retriever = vs.as_retriever(search_kwargs={"k": RECALL_K})
                docs = retriever.get_relevant_documents(content, callbacks=[self.cb_handler])

                # reranking
                docs = rank_docs(docs, content)
                docs = docs[:settings.retriever.top_k]

                result = []
                for doc in docs:
                    result.append(
                        {
                        "source": os.path.basename(doc.metadata.get('source', '')),
                        "content": doc.page_content
                        }
                    )
                return result
            return []
        except Exception as e:
            logger.error(f"Error from POST /search endpoint. Error details: {e}")

    def get_documents(self) -> List[str]:
        """Retrieves filenames stored in the vector store."""
        try:
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs:
                return get_docs_vectorstore_langchain(vs)
        except Exception as e:
            logger.error(f"Vectorstore not initialized. Error details: {e}")
        return []


    def delete_documents(self, filenames: List[str]):
        """Delete documents from the vector index."""
        try:
            vs = get_vectorstore(vectorstore, document_embedder)
            if vs:
                return del_docs_vectorstore_langchain(vs, filenames)
        except Exception as e:
            logger.error(f"Vectorstore not initialized. Error details: {e}")

    def get_processing_documents(self) -> List[str]:
        """Get the status of uploaded files."""
        global uploaded_files_status
        
        processing_filenames = []
        for filename, status in uploaded_files_status.items():
            if status == 0:
                processing_filenames.append(filename)
        return processing_filenames
