from langchain_openai import OpenAIEmbeddings
import lancedb
from apartment_listing_vector import ApartmentListingWithVector
from apartment_listing import ApartmentListing
from typing import List

class VectorDBManager:
    vector_db = lancedb.connect("~/.lancedb")
    apartments_table = vector_db.create_table("apartment_listing", schema=ApartmentListingWithVector, exist_ok= True)

    def save_apartments_in_vector_db(self, apartments: List[ApartmentListing]):
        # loop through or map generate models with vector and embeddings

        aptsWithVector = []
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=256,
            openai_api_base= "https://openai.vocareum.com/v1",
            openai_api_key= "voc-1462828482126677220028066b10710f0dff9.76617808"
        )

        for apt in apartments:
            if apt.description is not None and len(apt.description) > 0:
                vector = embeddings.embed_query(apt.description + apt.neighborhood_description)

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