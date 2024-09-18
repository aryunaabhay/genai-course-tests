from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import List
from langchain_openai import ChatOpenAI
from apartment_listing import ApartmentListing

llm = ChatOpenAI(
    model_name= "gpt-3.5-turbo",
    temperature=0.0,
    base_url = "https://openai.vocareum.com/v1",
    api_key = "voc-1462828482126677220028066b10710f0dff9.76617808"
)

class ListingsGenerator:
    def generateAparments(self) -> List[ApartmentListing]:
        # generate listings from llm with format and translate to pydantic models
        generate_listings_prompt = """Generate 10 apartment listings including the following information Neighborhood, Price, Bedrooms, Bathrooms, House Size, 
        Description, Neighborhood Description, make them in different sizes neighborhoods, prices and so on.
        {format_instructions}
        """
        parser = PydanticOutputParser(pydantic_object= ApartmentListing)
        format_instructions = parser.get_format_instructions()
        print(format_instructions)

        prompt_template = PromptTemplate(
            template= generate_listings_prompt,
            input_variables= ["format_instructions"],
        )

        query = prompt_template.format(format_instructions= format_instructions)
        output = llm.predict(query)
        listingsStrings = output.split('}')
        apartmentListings = []

        for i, listingStr in enumerate(listingsStrings):
            if(i != len(listingsStrings) - 1):
                listingFixed = listingStr + '}'
            aparment = parser.parse(listingFixed)
            apartmentListings.append(aparment)

        return apartmentListings