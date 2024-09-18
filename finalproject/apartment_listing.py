from pydantic import BaseModel

class ApartmentListing(BaseModel):
    neighborhood: str
    price: float
    bedrooms: int
    bathrooms: int
    size: int
    description: str
    neighborhood_description: str