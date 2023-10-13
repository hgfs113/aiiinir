import re
from typing import List, Optional

import pandas as pd
import PyPDF2
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


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

    def load(self, ) -> List[Document]:
        """Load from file path."""
        texts = []
        metadata = {"source": self.file_path}
        if self.file_path.endswith("csv"):
            df = pd.read_csv(self.file_path)
            for row in df.iterrows():
                text = f"Service: {row[1]['Service']}, Condition: {row[1]['Condition']}, Tariff: {row[1]['Tariff']}"
                texts.append(text)
            return [Document(page_content=text, metadata=metadata) for text in texts]
        elif self.file_path.endswith("pdf"):
            reader = PyPDF2.PdfReader(self.file_path)
            text = " ".join([page.extract_text() for page in reader.pages])
            text = text.replace('\xa0', ' ').replace('\n', ' ')
            pattern = r"\d+ из \d+"
            result = re.sub(pattern, "", text)
            texts = [row for row in result.split('.') if len(row) >= 10]
            return [Document(page_content=text, metadata=metadata) for text in texts]
        return []
