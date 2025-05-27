import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
        # print(text)
    return text

pdf_file = read_pdf("docs/CV_ES.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.create_documents([pdf_file])

len(documents)
for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("-" * 40)