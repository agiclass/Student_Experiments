from dotenv import load_dotenv
load_dotenv("api_keys.env")
import os
import re
import json
import requests
import weaviate
from tqdm import tqdm

def rrf(rankings, k=60):
    if not isinstance(rankings, list):
        raise ValueError("Rankings should be a list.")
    scores = dict()
    for ranking in rankings:
        if not ranking:  # 如果ranking为空，跳过它
            continue
        for i, doc in enumerate(ranking):
            if not isinstance(doc, dict):
                raise ValueError("Each item should be dict type.")
            doc_id = doc.get('hotel_id', None)
            if doc_id is None:
                raise ValueError("Each item should have 'hotel_id' key.")
            if doc_id not in scores:
                scores[doc_id] = (0, doc)
            scores[doc_id] = (scores[doc_id][0] + 1 / (k + i), doc)

    sorted_scores = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [item[1] for item in sorted_scores]


class HotelDB():
    def __init__(self, url="http://8.217.22.255:6500"):
        self.client = weaviate.Client(url=url,
          additional_headers={
              "X-OpenAI-Api-Key":os.getenv("OPENAI_API_KEY")
          }
        )

    def insert(self):
        self.client.schema.delete_class("Hotel")
        schema = {
          "classes": [
            {
              "class": "Hotel",
              "description": "hotel info",
              "properties": [
                {
                  "dataType": ["number"], 
                  "description": "id of hotel", 
                  "name": "hotel_id" 
                },
                {
                  "dataType": ["text"],
                  "description": "name of hotel",
                  "name": "_name", #分词过用于搜索的
                  "indexSearchable": True,
                  "tokenization": "whitespace",
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "type of hotel",
                  "name": "name",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "type of hotel",
                  "name": "type",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "address of hotel",
                  "name": "_address", #分词过用于搜索的
                  "indexSearchable": True,
                  "tokenization": "whitespace",
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "type of hotel",
                  "name": "address",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "nearby subway",
                  "name": "subway",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "phone of hotel",
                  "name": "phone",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                { 
                  "dataType": ["number"], 
                  "description": "price of hotel",   
                  "name": "price" 
                },
                { 
                  "dataType": ["number"], 
                  "description": "rating of hotel",
                  "name": "rating" 
                },
                {
                  "dataType": ["text"],
                  "description": "facilities provided",
                  "name": "facilities",
                  "indexSearchable": True,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": False }
                  },
                },
              ],
              "vectorizer": "text2vec-openai",
              "moduleConfig": {
                "text2vec-openai": {
                  "vectorizeClassName": False,
                  "model": "ada",
                  "modelVersion": "002",
                  "type": "text"
                },
              },
            }
          ]
        }

        self.client.schema.create(schema)

        url = 'https://raw.githubusercontent.com/agiclass/hotel-chatbot/main/data/hotel.json'
        if not os.path.exists('hotel.json'):
            print("Downloading file...")
            response = requests.get(url)
            with open('hotel.json', 'wb') as file:
                file.write(response.content)
            print("Download complete!")
        else:
            print("File already exists.")
        with open('hotel.json', 'r') as f:
            hotels = json.load(f)

        self.client.batch.configure(batch_size=4, dynamic=True)

        for hotel in tqdm(hotels):
            self.client.batch.add_data_object(
                data_object=hotel,
                class_name="Hotel",
                uuid=weaviate.util.generate_uuid5(hotel, "Hotel")
            )

        self.client.batch.flush()

    def search(self, dsl, name="Hotel", limit=1):
        # dsl中过滤掉None值
        dsl = {k: v for k, v in dsl.items() if v is not None}
        _limit = limit + 10 # 多搜10条，让取top `limit`条
        candidates = []
        output_fields = ["hotel_id","name","type","address","phone","subway","facilities","price","rating"]
        # ===================== assemble filters ========================= #
        filters = [{
            "path": ["price"],
            "operator": "GreaterThan",
            "valueNumber": 0,
        }]
        keys = [
            "type",
            "price_range_lower",
            "price_range_upper",
            "rating_range_lower",
            "rating_range_upper",
        ]
        if any(key in dsl for key in keys):
            if "type" in dsl:
                filters.append(
                    {
                        "path": ["type"],
                        "operator": "Equal",
                        "valueString": dsl["type"],
                    }
                )
            if "price_range_lower" in dsl:
                filters.append(
                    {
                        "path": ["price"],
                        "operator": "GreaterThan",
                        "valueNumber": dsl["price_range_lower"],
                    }
                )
            if "price_range_upper" in dsl:
                filters.append(
                    {
                        "path": ["price"],
                        "operator": "LessThan",
                        "valueNumber": dsl["price_range_upper"],
                    }
                )
            if "rating_range_lower" in dsl:
                filters.append(
                    {
                        "path": ["rating"],
                        "operator": "GreaterThan",
                        "valueNumber": dsl["rating_range_lower"],
                    }
                )
            if "rating_range_upper" in dsl:
                filters.append(
                    {
                        "path": ["rating"],
                        "operator": "LessThan",
                        "valueNumber": dsl["rating_range_upper"],
                    }
                )
        if (len(filters)) == 1:
            filters = filters[0]
        elif len(filters) > 1:
            filters = {"operator": "And", "operands": filters}
        # ===================== vector search ============================= #
        if "facilities" in dsl:
            query = self.client.query.get(name, output_fields)
            query = query.with_near_text(
                {"concepts": [f'酒店提供:{";".join(dsl["facilities"])}']}
            )
            if filters:
                query = query.with_where(filters)
            query = query.with_limit(_limit)
            result = query.do()
            candidates = rrf([candidates, result["data"]["Get"][name]])
        # ===================== keyword search ============================ #
        if "name" in dsl:
            text = " ".join(re.findall(r"[\dA-Za-z\-]+|\w", dsl["name"]))
            query = self.client.query.get(name, output_fields)
            query = query.with_bm25(query=text, properties=["_name"])
            if filters:
                query = query.with_where(filters)
            query = query.with_limit(_limit)
            result = query.do()
            candidates = rrf([candidates, result["data"]["Get"][name]])
        if "address" in dsl:
            text = " ".join(re.findall(r"[\dA-Za-z\-]+|\w", dsl["address"]))
            query = self.client.query.get(name, output_fields)
            query = query.with_bm25(query=text, properties=["_address"])
            if filters:
                query = query.with_where(filters)
            query = query.with_limit(_limit)
            result = query.do()
            candidates = rrf([candidates, result["data"]["Get"][name]])
        # ====================== condition search ========================== #
        if not candidates:
            query = self.client.query.get(name, output_fields)
            if filters:
                query = query.with_where(filters)
            query = query.with_limit(_limit)
            result = query.do()
            candidates = result["data"]["Get"][name]
        # ========================== sort ================================= #
        if "sort.slot" in dsl:
            if dsl["sort.ordering"] == "descend":
                candidates = sorted(
                    candidates, key=lambda x: x[dsl["sort.slot"]], reverse=True
                )
            else:
                candidates = sorted(
                    candidates, key=lambda x: x[dsl["sort.slot"]]
                )
        
        if "name" in dsl:
          final = []
          for r in candidates:
              if all(char in r['name'] for char in dsl['name']):
                  final.append(r)
          candidates = final
        
        if len(candidates) > limit:
            candidates = candidates[:limit]
        
        return candidates


if __name__ == "__main__":
    db = HotelDB()
    result = db.search({'facilities':['打麻将']}, limit=3)
    print(json.dumps(result,ensure_ascii=False))
