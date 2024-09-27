from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from apartment_listing import ApartmentListing
from llm import llm

class ApartmentQuery:
    def create_query_obj_from_conversation(self, bedrooms, bathrooms, neighborhood_description, price, size, others):
        memory = ConversationBufferMemory()
        memory.save_context(
            {"output": "and what about number of bedrooms"}, 
            {"input": bedrooms}
            )
        memory.save_context(
            {"output": "and what about number of bathrooms"}, 
            {"input": bathrooms}
            )
        memory.save_context(
            {"output": "describe your ideal neighborhood"}, 
            {"input": neighborhood_description}
            )
        memory.save_context(
            {"output": "and what about price wise"}, 
            {"input": price}
            )
        memory.save_context(
            {"output": "and what about size?"}, 
            {"input": size}
            )
        memory.save_context(
            {"output": "and what about other details"}, 
            {"input": others}
            )

        conversation_chain = ConversationChain(llm= llm, memory= memory, verbose= True)
        parser = PydanticOutputParser(pydantic_object= ApartmentListing)
        format_instructions = parser.get_format_instructions()

        prompt_template = PromptTemplate(
                    template= "make a description containing user preferences and format all preferences including new description them as json object, {format_instructions}",
                    input_variables= ["format_instructions"],
                )

        query = prompt_template.format(format_instructions= format_instructions)
        output = conversation_chain.predict(input=query)
        aparment = parser.parse(output)
        return aparment
