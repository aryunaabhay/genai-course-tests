
import lancedb
from apartment_listing_vector import ApartmentListingWithVector
from apartment_listing import ApartmentListing
from typing import List
from llm import embeddings

class VectorDBManager:
    apartments_table_name = "apartment_listing"
    
    def __init__(self):
        self.vector_db = lancedb.connect("~/.lancedb")
        self.apartments_table = self.vector_db.create_table(self.apartments_table_name, schema=ApartmentListingWithVector, exist_ok= True)
    
    def check_if_apartments_generated(self):
        table_exists = self.apartments_table_name in self.vector_db.table_names()
        return table_exists
    
    def drop_apts_table(self):
        self.vector_db.drop_table(self.apartments_table_name)
    
    def open_apts_table(self):
        self.apartments_table = self.vector_db.open_table(self.apartments_table_name)
    
    def save_apartments_in_vector_db(self, apartments: List[ApartmentListing]):
        # loop through or map generate models with vector and embeddings
        aptsWithVector = []

        for apt in apartments:
            if apt.description is not None and len(apt.description) > 0:
                vector = embeddings.embed_query(apt.description)

                aptWithVector = ApartmentListingWithVector(
                    neighborhood= apt.neighborhood,
                    price= apt.price,
                    bedrooms= apt.bedrooms,
                    bathrooms= apt.bathrooms,
                    size= apt.size,
                    description= apt.description,
                    neighborhood_description= apt.neighborhood_description,
                    vector= vector
                )
                aptsWithVector.append(aptWithVector)

        self.apartments_table.add(aptsWithVector)