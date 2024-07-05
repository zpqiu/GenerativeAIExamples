import os
import re
import json
import logging

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
if NVIDIA_API_KEY is None:
    raise ValueError('Please set the environment variable NVIDIA_API_KEY')


def reorder_elements_for_double_columns(elements):
    # PDF是左右双栏排列的，需要对元素进行排序
    # 统计最大的 listitem 和 narrative text 的宽度，然后根据宽度判断是左栏还是右栏
    # 顺便统计最小的 Title, listItem, narrative text, image, table 的x坐标，作为左栏的起始坐标
    max_width = 0
    min_x = 10000
    for element in elements:
        if element['type'] == 'ListItem' or element['type'] == 'NarrativeText':
            width = float(element['bbox'][2][0]) - float(element['bbox'][0][0])
            if width > max_width:
                max_width = width

        if element['type'] == 'Title' or element['type'] == 'ListItem' or \
            element['type'] == 'NarrativeText' or element['type'] == 'Image' or element['type'] == 'Table':
            x = float(element['bbox'][0][0])
            if x < min_x:
                min_x = x

    # elements 本来就是按照从上到下排列的，我们只需要拆分成左右两栏即可
    left_elements = []
    right_elements = []
    for element in elements:
        # 如果 element 的左上角点的 x 坐标小于 min_x + max_width，那么它属于左栏
        if float(element['bbox'][0][0]) < min_x + max_width:
            left_elements.append(element)
        else:
            right_elements.append(element)

    return left_elements + right_elements


prompt_template = """
下面的json是从一个pdf文件中提取的元素，包括文本、表格和图片。文本元素包含了文本内容、类型，图片和表格元素包含了类型、原始 pdf 中截图的路径信息。
你需要根据这些 json 元素，将其转换为更易于Markdown格式。

注意：
- 保留正文、图片和表格
- pdf 解析时文本内容中会出现多余的空格和符号，需要去除
- 对于和正文无关的 pdf 的页眉、页脚、页码等信息，可以忽略

```jsonl
{elements}
```

请输出转换后的 Markdown 格式，使用```markdown```包裹输出的文本。
"""

from openai import OpenAI

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = NVIDIA_API_KEY
)

def llm_refine(prompt):
    completion = client.chat.completions.create(
        model="google/gemma-2-27b-it",
        messages=[{"role":"user","content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )

    ret = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            ret += chunk.choices[0].delta.content

    md_block = re.search(r'```markdown\n(.*)\n```', ret, re.DOTALL)
    if md_block is None:
        return ret
    else:
        return md_block.group(1)


def refine_by_page(elements, is_double_columns=True):
    refined_elements = []
    for element in elements:
        if element['type'] == 'Table' or element['type'] == 'Image':
            refined_elements.append({
                'type': element['type'],
                'bbox': element['metadata']['coordinates']['points'],
                'image_path': element['metadata']['image_path']
            })
        else:
            refined_elements.append({
                'type': element['type'],
                'bbox': element['metadata']['coordinates']['points'],
                'text': element['text'],
            })

    if is_double_columns:
        refined_elements = reorder_elements_for_double_columns(refined_elements)

    # 去掉所有的 bbox 属性值
    for element in refined_elements:
        element.pop('bbox', None)

    json_blocks = ""
    for element in refined_elements:
        json_blocks += json.dumps(element, ensure_ascii=False) + '\n'

    prompt = prompt_template.format(elements=json_blocks.strip())

    try:
        md_text = llm_refine(prompt)
    except Exception as e:
        logger.error(f"Failed to format chunk: {e}")
        md_text = "\n".join([element['text'] for element in refined_elements 
                             if element['type'] != 'Image' and element['type'] != 'Table'])
        # logger.info(f"Failed to format chunk: {md_text}")

    return md_text
