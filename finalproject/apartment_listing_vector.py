from lancedb.pydantic import Vector, LanceModel

class ApartmentListingWithVector(LanceModel):
    vector: Vector(256)
    neighborhood: str
    price: float
    bedrooms: int
    bathrooms: int
    size: int
    description: str
    neighborhood_description: str