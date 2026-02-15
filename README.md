WORKFLOW:
1. First, I built a tokenizer(ingest.py). In this layer when a pdf is given, it is extracted page by page into raw text data for processing in further steps. for this i have used pypdf lib from which i extract the text.


2. Now this raw data is divided into chunks. I am using Langchain. A framework that helps me provide a better context of what prompt i am about to give to the model in the further steps.

