from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

base_url = "https://openai.vocareum.com/v1"
api_key = "voc-1462828482126677220028066b10710f0dff9.76617808"

llm = ChatOpenAI(
    model_name= "gpt-3.5-turbo",
    temperature=0.0,
    base_url = base_url,
    api_key = api_key
)

embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=256,
            openai_api_base= base_url,
            openai_api_key= api_key
        )