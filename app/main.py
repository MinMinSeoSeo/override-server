import os
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import List
import humps
import pandas as pd
import os


app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=humps.camelize,
        populate_by_name=True
    )


class GroupType(str, Enum):
    family = "family"
    friends = "friends"
    couple = "couple"
    solo = "solo"


class AgeGroupStatus(str, Enum):
    elderly = "elderly"
    child = "child"
    both = "both"
    none = "none"


class DifficultyLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AttractionRecommendRequest(CamelCaseModel):
    attraction_count: int = Field(..., ge=1, le=5)
    group_type: GroupType
    age_group_status: AgeGroupStatus
    difficulty_levels: List[DifficultyLevel]
    theme_tags: List[str]


class Attraction(CamelCaseModel):
    name: str
    image_url: str


class AttractionGroup(CamelCaseModel):
    attractions: List[Attraction]


class AttractionRecommendResponse(CamelCaseModel):
    attractionGroups: List[AttractionGroup]


def extract_random_items(source_list, count):
    extracted_items = []

    for _ in range(count):
        if source_list:
            selected_item = random.choice(source_list)
            extracted_items.append(selected_item)
            source_list.remove(selected_item)

    return extracted_items


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/attractions/recommendations")
async def get_attraction_recommendations(request: AttractionRecommendRequest):
    file_path = './app/data/attractions.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return HTTPException(status_code=404, detail="Data file not found")

    allAttractionList = [
        Attraction(name=row['name'], image_url=row['image_url'])
        for _, row in df.iterrows()
    ]

    print(len(allAttractionList))

    attractionGroups = []

    for _ in range(5):
        if len(allAttractionList) < request.attraction_count:
            break

        selected_items = extract_random_items(
            allAttractionList, request.attraction_count)

        attractionGroups.append(AttractionGroup(
            attractions=selected_items))

    return AttractionRecommendResponse(attractionGroups=attractionGroups)
