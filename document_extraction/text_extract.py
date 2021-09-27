from typing import Dict, Any, List
import pdfplumber




def extract_text_line(text_line_dict : List[Dict[str,Any]]) -> List[List]:
    page_text, line_block = [], []
    prev_top = 0
    for idx, text_line in enumerate(text_line_dict):
        if idx !=0:
            section_break_size = text_line['top'] - prev_top
            if section_break_size < 1:
                line_block.append(text_line['text'])
            else:
                page_text.append(line_block)
                line_block = [text_line['text']]

        prev_top = text_line['top']
    return page_text




with pdfplumber.open(f'data/pdf/astro-ph0001004.pdf') as pdf:
    for idx, page in enumerate(pdf.pages):
        line = extract_text_line(page.extract_words())
        print(line)



