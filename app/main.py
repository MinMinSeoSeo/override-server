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
    description: str
    location: str
    difficulty: DifficultyLevel
    usage_info: str
    concept: str

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

def attraction_filter(attraction_row, group_type, difficulty_levels):
    if group_type not in attraction_row['companion_type']+',solo':
        return False
    if attraction_row['difficulty'] not in difficulty_levels:
        return False
    return True

def score_estimator(attraction_row, age_group_status, theme_tags):
    if age_group_status == 'both':
        score = 0.5 * (attraction_row['senior_friendly_score'] + attraction_row['child_friendly_score'])
    elif age_group_status == 'elderly':
        score = attraction_row['senior_friendly_score']
    elif age_group_status == 'child':
        score = attraction_row['child_friendly_score']
    else:
        score = 0.5

    attraction_theme = attraction_row['theme'].split(',')
    include_theme_count = 0
    for t in theme_tags:
        if t in attraction_theme:
            include_theme_count += 1
    theme_score = include_theme_count / len(theme_tags)
    score = 0.5*(score + theme_score)

    return score

def backtrack(all_attraction_list, start, path, current_score, all_combinations, attraction_count):
    if len(path) == attraction_count:
        all_combinations.append((path.copy(), current_score))
        return
    for i in range(start, len(all_attraction_list)):
        attraction = all_attraction_list[i]
        path.append(attraction)
        backtrack(
            all_attraction_list,
            i + 1,
            path,
            current_score + attraction['score'],
            all_combinations,
            attraction_count
        )
        path.pop()

def to_attraction_form(recommended_combinations):
    attraction_groups = []
    for combo, _ in recommended_combinations:
        attractions = []
        for attr in combo:
            attractions.append(Attraction(
                name = attr['name'],
                image_url = attr['image_url'],
                description = attr['description'],
                location = attr['location'],
                difficulty = attr['difficulty'],
                usage_info = attr['usage_info'],
                concept = attr['concept']
            ))
        attraction_groups.append(AttractionGroup(attractions=attractions))
    return attraction_groups

@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/attractions/recommendations", response_model=AttractionRecommendResponse)
async def recommend_attractions(request: AttractionRecommendRequest):
    file_path = './app/data/attractions.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return HTTPException(status_code=404, detail="Data file not found (check the attractions.csv)")

    filtered_df = df[df.apply(lambda row: attraction_filter(row, request.group_type, request.difficulty_levels), axis=1)]

    allAttractionList = []
    for _, row in filtered_df.iterrows():
        score = score_estimator(row, request.age_group_status, request.theme_tags)
        attraction = {
            'name': row['name'],
            'image_url': row['image_url'],
            'description': row['description'],
            'location': row['location'],
            'difficulty': row['difficulty'],
            'usage_info': row['usage_info'],
            'concept': row['concept_tags'],
            'score': score
        }
        allAttractionList.append(attraction)

    print(len(allAttractionList))

    all_combinations = []
    backtrack(allAttractionList, 0, [], 0, all_combinations, request.attraction_count)

    option_number = 5 #화면에 표시할 놀이기구 조합의 개수 (더보기 눌렀을 때 기준 5개)
    all_combinations_sorted = sorted(all_combinations, key=lambda x: x[1], reverse=True)
    recommended_combinations = all_combinations_sorted[:option_number]
    
    """
    print(f"{option_number}개의 놀이기구 조합:")
    for idx, (combo, score) in enumerate(recommended_combinations, 1):
        attraction_names = [attr['name'] for attr in combo]
        print(f"\n조합 {idx}: (총 점수: {score:.2f})")
        for name in attraction_names:
            print(f" - {name}")
    """

    attractionGroups = to_attraction_form(recommended_combinations)

    return AttractionRecommendResponse(attractionGroups=attractionGroups)
