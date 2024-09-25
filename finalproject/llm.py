from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name= "gpt-3.5-turbo",
    temperature=0.0,
    base_url = "https://openai.vocareum.com/v1",
    api_key = "voc-1462828482126677220028066b10710f0dff9.76617808"
)