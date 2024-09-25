from listings_generator import ListingsGenerator
from vector_db_manager import VectorDBManager
from apartment_listing_vector import ApartmentListingWithVector
from vector_db_manager import embeddings
from conversation_query import ApartmentQuery

#apts = ListingsGenerator().generateAparments()
vector_db_manager = VectorDBManager()
#vector_db_manager.save_apartments_in_vector_db(apts)

apartment_query = ApartmentQuery().create_query_obj_from_conversation()
vector_search = embeddings.embed_query(apartment_query.description + apartment_query.neighborhood_description)
search_result = vector_db_manager.apartments_table.search(vector_search).limit(1).to_pydantic(ApartmentListingWithVector)
print(search_result)

# another query to llm  recreate descriptions from search of this apartment highlihting the user preference without losing factual integrity
