from listings_generator import ListingsGenerator
from vector_db_manager import VectorDBManager

import numpy as np

apts = ListingsGenerator().generateAparments()
vector_db_manager = VectorDBManager()
vector_db_manager.save_apartments_in_vector_db(apts)

#langchain 
random_vector = np.random.rand(256)
print(vector_db_manager.apartments_table.search(random_vector).limit(2))

# questions and answers to colelct user preferences about listings
# structure user preference and create query for search in vector db

# search semantically 
# each result augment with llm to highlight the thing user prefers
# maintaining factual integrity