from typing import List, Optional

import pandas as pd
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


MIN_SETNTENCE_LEN = 8


class MyDocLoader(BaseLoader):
    def __init__(
            self,
            file_path: str,
            encoding: Optional[str] = None,
            autodetect_encoding: bool = False,
        ):
            """Initialize with file path."""
            self.file_path = file_path
            self.encoding = encoding
            self.autodetect_encoding = autodetect_encoding

    def load(self,) -> List[Document]:
        """Load from file path."""
        result = []
        metadata = {"source": self.file_path}
        if self.file_path.endswith("csv"):
            df = pd.read_csv(self.file_path)
            for _, item in df.iterrows():
                text = f"Сервис: {item['Service']}, Условие: {item['Condition']}, Тарифф: {item['Tariff']}"
                result.append(Document(page_content=text, metadata=metadata))
        elif self.file_path.endswith("pdf"):
            reader = PdfReader(self.file_path)
            text = " ".join([page.extract_text() for page in reader.pages])
            text = text.replace('\xa0', ' ')
            text = text.replace('\n', ' ')
            for sentence in text.split('.'):
                if len(sentence) < MIN_SETNTENCE_LEN:
                    continue
                result.append(Document(page_content=sentence, metadata=metadata))
        return result
