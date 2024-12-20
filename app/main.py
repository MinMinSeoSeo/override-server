import os
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import List
import humps
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv
import json

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://override-amusement-park.com",
    "https://www.override-amusement-park.com",
    "https://override-amusement-park.vercel.app",
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


class RecommendTestRequest(CamelCaseModel):
    query: str


class MessageResponse(CamelCaseModel):
    message: str


class EmbeddingRecommendResponse(CamelCaseModel):
    name: str
    score: float

def extract_random_items(source_list, count): #사용 X
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

def get_embedding_scores(theme_tags):
    if not theme_tags:
        result = {}
        return result
    else:
        result = get_recommend_scores(theme_tags[0])
    for tag in theme_tags[1:]:
        temp = get_recommend_scores(tag)
        for attr_name in temp:
            result[attr_name] += temp[attr_name]
    for attr_name in result:
        result[attr_name] /= len(theme_tags)
    #print(json.dumps(result, indent=4))
    return result

def score_estimator(attraction_row, age_group_status, theme_score):
    if age_group_status == 'both':
        score = 0.5 * \
            (attraction_row['senior_friendly_score'] +
             attraction_row['child_friendly_score'])
    elif age_group_status == 'elderly':
        score = attraction_row['senior_friendly_score']
    elif age_group_status == 'child':
        score = attraction_row['child_friendly_score']
    else:
        score = 0.5

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

def select_combinations(sorted_combinations, option_number, max_overlap):
    recommend_combinations = []
    for combo in sorted_combinations:
        max_duplicated_count = 0
        for existing_combo in recommend_combinations:
            duplicated_count = 0
            for attr in combo[0]:
                if attr in existing_combo[0]:
                    duplicated_count += 1
            if duplicated_count > max_duplicated_count:
                max_duplicated_count = duplicated_count
        
        if max_duplicated_count <= max_overlap:
            random.shuffle(combo[0])
            recommend_combinations.append(combo)
        if len(recommend_combinations) == option_number:
            break
    return recommend_combinations

def to_attraction_form(recommended_combinations):
    attraction_groups = []
    for combo, _ in recommended_combinations:
        attractions = []
        for attr in combo:
            attractions.append(Attraction(
                name=attr['name'],
                image_url=attr['image_url'],
                description=attr['description'],
                location=attr['location'],
                difficulty=attr['difficulty'],
                usage_info=attr['usage_info'],
                concept=attr['concept']
            ))
        attraction_groups.append(AttractionGroup(attractions=attractions))
    return attraction_groups


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


def get_recommend_scores(query, test=False):
    query_embedding = np.array(get_embedding(query))
    query_embedding = query_embedding.reshape(1, -1)

    with open("./app/data/embeddings.json", "r") as f:
        embeddings = json.load(f)

    attraction_names = list(embeddings.keys())
    attraction_embeddings = np.array(list(embeddings.values()))

    similarities = cosine_similarity(query_embedding, attraction_embeddings)[0]
    #ranked_indices = np.argsort(similarities)[::-1]

    #recommendations = [(attraction_names[i], float(similarities[i])) for i in ranked_indices[:]]
    recommendations = [(attraction_names[i], float(similarities[i])) for i in range(len(similarities[:]))]

    if test:
        result = [
            {
                "name": name,
                "score": score
            } for name, score in recommendations
        ]
    else:
        result = {}
        for name, score in recommendations:
            result[name] = score

    return result

@app.get("/", response_model=MessageResponse)
def read_root():
    return {"message": "Hello World"}

@app.post("/attractions/recommendations", response_model=AttractionRecommendResponse)
async def recommend_attractions(request: AttractionRecommendRequest):
    file_path = './app/data/attractions.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        raise HTTPException(status_code=404, detail="Data file not found (check the attractions.csv)") # 수정 1: return -> raise

    filtered_df = df[df.apply(lambda row: attraction_filter(
        row, request.group_type, request.difficulty_levels), axis=1)]
    
    embedding_scores = get_embedding_scores(request.theme_tags)

    allAttractionList = []
    for _, row in filtered_df.iterrows():
        if row['name'] not in embedding_scores: embedding_scores[row['name']] = 0
        score = score_estimator(row, request.age_group_status, embedding_scores[row['name']])
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

    # print('~-'*50, len(allAttractionList))

    all_combinations = []
    backtrack(allAttractionList, 0, [], 0,
              all_combinations, request.attraction_count)

    option_number = 5 #화면에 표시할 놀이기구 조합의 개수 (더보기 눌렀을 때 기준 5개)
    max_overlap = 1 #추천 조합 간 중복되는 놀이기구의 최대 개수
    all_combinations_sorted = sorted(all_combinations, key=lambda x: x[1], reverse=True)
    recommended_combinations = select_combinations(all_combinations_sorted, option_number, max_overlap)
    
    """
    print(f"{option_number}개의 놀이기구 조합:")
    for idx, (combo, score) in enumerate(recommended_combinations, 1):
        attraction_names = [attr['name'] for attr in combo]
        print(f"\n조합 {idx}: (총 점수: {score:.2f})")
        for name in attraction_names:
            print(f" - {name} ({combo[0]['difficulty']})")
    """

    attractionGroups = to_attraction_form(recommended_combinations)

    return AttractionRecommendResponse(attractionGroups=attractionGroups)


@app.post('/embeddings/update', response_model=MessageResponse)
async def update_embeddings():
    df = pd.read_csv('./app/data/attractions.csv')

    attraction_names = df['name'].tolist()
    attraction_descriptions = df['description'].tolist()
    attraction_concepts = df['concept_tags'].tolist()

    embeddings = {}

    for name, description, concept in zip(attraction_names, attraction_descriptions, attraction_concepts):
        embedding = get_embedding(f"{name} {description} {concept}")
        embeddings[name] = embedding

    with open("./app/data/embeddings.json", "w") as f:
        json.dump(embeddings, f)

    return {"message": "Embeddings Updated!"}


@app.post('/embeddings/recommend', response_model=List[EmbeddingRecommendResponse])
async def recommend_test(request: RecommendTestRequest):
    result = get_recommend_scores(request.query, test=True)

    return result
