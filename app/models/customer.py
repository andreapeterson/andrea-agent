from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PetSpecies(str, Enum):
    DOG = "dog"
    CAT = "cat"


class Pet(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    pet_id: str
    name: str
    species: PetSpecies


class Customer(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str
    first_name: str
    last_name: str
    phone_number: str
    pets: list[Pet] = Field(default_factory=list) #if pets is emtpy, creates empty list
