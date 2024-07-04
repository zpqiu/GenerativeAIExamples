"""
基于 FastAPI 库写一个 Web 服务端，它只有一个接口：
- 一个接口接收两个路径字符串，表示一个 pdf 文件的路径，和解析后的文件夹（包含了 jsonl 和 pdf 中的图片）路径，返回一个状态码，表示解析是否成功
"""
import os
import json
import shutil
import logging
from typing import List

from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from unstructured.partition.pdf import partition_pdf

from pdfparser.llm_refine import refine_by_page

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Path(BaseModel):
    pdf_path: str
    output_path: str


@app.post("/parse_pdf")
async def parse_pdf(path: Path) -> JSONResponse:
    if not os.path.exists(path.pdf_path):
        return JSONResponse(status_code=400, content={"message": "pdf file not found"})
    if not os.path.exists(path.output_path):
        return JSONResponse(status_code=400, content={"message": "output path not found"})
    if not path.pdf_path.endswith('.pdf'):
        return JSONResponse(status_code=400, content={"message": "pdf file should end with .pdf"})
    
    image_folder_path = os.path.join(path.output_path, "images")
    pdf_filename = os.path.basename(path.pdf_path)
    md_path = os.path.join(path.output_path, "doc.md")
    os.makedirs(image_folder_path, exist_ok=True)

    elements = partition_pdf(
        filename=path.pdf_path,                                 # mandatory
        # strategy="hi_res",                                     # mandatory to use ``hi_res`` strategy
        extract_images_in_pdf=True,                            # mandatory to set as ``True``
        extract_image_block_types=["Image", "Table"],          # optional
        extract_image_block_to_payload=False,                  # optional
        extract_image_block_output_dir=image_folder_path,  # optional - only works when ``extract_image_block_to_payload=False``
    )
    logger.info(f"[File][{pdf_filename}]Extracted {len(elements)} elements from pdf file")
    elements = [e.to_dict() for e in elements]

    max_page_no = elements[-1]['metadata']['page_number']
    # change element list to list of list according the page number
    elements_by_page = [[] for _ in range(max_page_no + 1)]
    for element in elements:
        elements_by_page[element['metadata']['page_number']].append(element)

    for i in range(max_page_no + 1):
        if (i + 1) % 20 == 0:
            logger.info(f"[File][{pdf_filename}]Processing page {i + 1} / {max_page_no}...")
        if len(elements_by_page[i]) == 0:
            continue

        md = refine_by_page(elements_by_page[i])
        with open(md_path, 'a') as f:
            f.write(md)

    return JSONResponse(status_code=200, content={"message": "success"})


# Run the server with uvicorn
# uvicorn server:app --reload

# test client
# import requests
#
# req = requests.post("http://localhost:8088/parse_pdf", 
#                     json={"pdf_path": "/docs/test.pdf", "output_path": "output"})
# print(req.text)
# print(req.status_code)