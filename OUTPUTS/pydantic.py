# pydantic is a data validation and data parsing library 
# It ensures that data is in correct format  , structure 

from pydantic import BaseModel

class Student(BaseModel):
    name : str 
