from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from apartment_listing import ApartmentListing
from llm import llm

class ApartmentQuery:
    def create_query_obj_from_conversation(self):
        memory = ConversationBufferMemory()
        memory.save_context(
            {"output": "what are your preferences in tems of neighborhood"}, 
            {"input": "I would love a neighborhood close to a park, and a good amount of trees around it in adittion to that will like for the neighborhood to have near by access to good restaurants"}
            )
        memory.save_context(
            {"output": "and what about number of bedrooms"}, 
            {"input": "2 bedrooms"}
            )
        memory.save_context(
            {"output": "and what about number of bathrooms"}, 
            {"input": "1 bathroom"}
            )
        memory.save_context(
            {"output": "and what about price wise"}, 
            {"input": "1500 usd aproximately"}
            )
        memory.save_context(
            {"output": "and what about size?"}, 
            {"input": "60 squarefoot"}
            )

        conversation_chain = ConversationChain(llm= llm, memory= memory, verbose= True)
        parser = PydanticOutputParser(pydantic_object= ApartmentListing)
        format_instructions = parser.get_format_instructions()

        prompt_template = PromptTemplate(
                    template= "take user answers and format them as json, {format_instructions}",
                    input_variables= ["format_instructions"],
                )

        query = prompt_template.format(format_instructions= format_instructions)
        output = conversation_chain.predict(input=query)
        aparment = parser.parse(output)
        return aparment
