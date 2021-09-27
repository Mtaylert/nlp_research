import pandas as pd
import pdfplumber
import time


start = time.time()
for doc in [1,2]:
    with pdfplumber.open(f'data/doc{doc}.pdf') as pdf:
        page_range = len(pdf.pages)
        for pg in range(page_range):
            pg_open = pdf.pages[pg]
end = time.time()

total = end-start
print(total)