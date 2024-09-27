from listings_generator import ListingsGenerator
from vector_db_manager import VectorDBManager
from apartment_listing_vector import ApartmentListingWithVector
from llm import embeddings
from conversation_query import ApartmentQuery
from llm import llm

vector_db = VectorDBManager()
if(vector_db.check_if_apartments_generated):
    vector_db.open_apts_table()
else:
    generated_apts = ListingsGenerator().generate_aparments()
    print(generated_apts)
    vector_db.save_apartments_in_vector_db(generated_apts)

apartment_query = ApartmentQuery().create_query_obj_from_conversation(
    bedrooms= "2 bedrooms",
    bathrooms= "1 bathroom",
    neighborhood_description= "close to a park, and a good amount of trees around it in adition to that will like to have near by access to good restaurants",
    price= "price 1500 USD aproximately",
    size= "600 square foot",
    others= "the apartment should have good style modern kitchen and modern style"
)
vector_search = embeddings.embed_query(apartment_query.description + apartment_query.neighborhood_description)
matches = vector_db.apartments_table.search(vector_search).limit(5).to_pydantic(ApartmentListingWithVector)

matches_augmented = []
for match in matches:
    augmented_description = llm.predict(f"""
                        recreate the apartment description highlighting the user preferences without chacing the features of the apartment
                        apartment description: {match.description}
                        user preferences : {apartment_query.description}
                        """)
    matches_augmented.append(augmented_description)

print(matches_augmented)
