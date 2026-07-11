import truststore
truststore.inject_into_ssl()
from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
existing = [idx.name for idx in pc.list_indexes()]
print(f"Existing indexes: {existing}")

name = "doc-helper-index"
if name in existing:
    print(f'Index "{name}" already exists!')
else:
    pc.create_index(
        name=name,
        dimension=1536,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    )
    print(f'Created index: {name} (1536 dims, cosine, serverless)')
