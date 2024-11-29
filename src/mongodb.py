from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.client_session import ClientSession

from bson.binary import Binary, BinaryVectorDtype
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
import numpy as np
import pymongoarrow as pma
from pymongoarrow.monkey import patch_all
import time
from typing import List, Dict, Union, Literal
import random
import uuid
import torch
import copy

@torch.compile()
def get_cosim(input1:torch.Tensor, input2:torch.Tensor)->torch.Tensor:
    """
    input1,input2: torch.Tensor shape (B_{i}, 512)
    """
    input2 = torch.cat([input2, input2, input2, input2, input2],dim = 0)
    norm_1 = torch.unsqueeze(torch.norm(input1, dim =1), dim = 1) 
    norm_2 = torch.unsqueeze(torch.norm(input2, dim =1), dim = 0)
    length_mul_matrix = 1/torch.mul(norm_1, norm_2)

    dot_product = torch.matmul(input1, torch.transpose(input2,0,1))

    dot_product_score = torch.mul(dot_product, length_mul_matrix)

    return torch.max(dot_product_score)

class UserEmbeddingSearch(object):
    def __init__(self, user_embedding_list: List[Dict[str, Union[str, list]]]):
        device = torch.device('cpu')
        self.master_data = [
            {
                'user_name': ele['user_name'],
                'embeddings': torch.tensor(ele['embeddings'], 
                                            dtype = torch.float32, 
                                            device = device)
            }
            for ele in user_embedding_list
        ]

    def search(self, query_embedding: torch.Tensor):
        start_time = time.time()
        score_dict = {
            user_dict['user_name']: get_cosim(user_dict['embeddings'],
                                                    query_embedding.cpu())
            for user_dict in self.master_data
        }
        print(f"processing time: {time.time() - start_time}")

        sorted_score_dict = {
            k:v.item() for k,v in score_dict.items()
        }
        sorted_score_dict = {k:v for k,v in \
            sorted(score_dict.items(), key=lambda item: item[1])
        }
        return list(sorted_score_dict.keys())[-1]


def generate_bson_vector(vector):
   return Binary.from_vector(vector, BinaryVectorDtype.FLOAT32)

class Mongo_Handler(object):
    def __init__(self,
                 master_config: dict,
                 ini_push: bool= False, 
                 init_data = None
                 ) -> None:

        mongoDB_username = quote_plus(master_config['mongodb_name'])
        mongoDB_pass = quote_plus(master_config['mongodb_pass'])

        uri = "mongodb+srv://"+ \
                mongoDB_username+":"+mongoDB_pass+\
                "@cluster0.pjmgwbo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        # Create a new client and connect to the server
        self.client = MongoClient(uri, 
                                  server_api=ServerApi(version='1'))
        # patch_all()
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")

        except Exception as e:
            print(e)

        if ini_push:
            assert init_data is not None
            with self.client.start_session() as session:
                with session.start_transaction():
                    self._ini_insert(init_data= copy.deepcopy(init_data))
                    self.UserEmbeddingSearch = UserEmbeddingSearch(init_data)
                    # vector_dim = int(master_config['vector_emebd_dim'])
                    # self._make_search_index(vector_dim = vector_dim)
        # else:
            # database = self.client["Storage"]
            # collection = database["Infor"]

            # start_time = time.time()
            # all_docs = collection.find(filter={})
            # self.UserEmbeddingSearch = UserEmbeddingSearch(all_docs)

    def close(self):
        self.client.close()

    def _ini_insert(self, 
                    init_data: List[Dict[str,Union[str,np.ndarray]]]):
        """
        Run for first time setup
        [
            {
                'embeddings': embedding dtype: torch.Tensor,
                'user_name': some_name
            }
        ]
        """
        database = self.client["Storage"]
        try:
            collection = database.create_collection(name= 'Infor')
            # for i in range(len(init_data)):
            #     init_data[i]['embedding'] = generate_bson_vector(init_data[i]['embedding'])

            for i in range(len(init_data)):
                init_data[i]['embeddings'] = init_data[i]['embeddings'].cpu().tolist()

            collection.insert_many(init_data)
        except Exception as e:
            print(e)

    def _make_search_index(self, vector_dim:int):
        """ Run for first time setup """
        database = self.client["Storage"]
        collection = database["Infor"]

        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": vector_dim,
                    "similarity": "cosine"   # euclidean | cosine | dotProduct
                }
                ]
            },
            name="vector_index",
            type="vectorSearch",
        )
        result = collection.create_search_index(model=search_index_model)

        print("New search index named " + result + " is building.")
        # Wait for initial sync to complete
        print("Polling to check if the index is ready. This may take up to a minute.")
        predicate=None
        if predicate is None:
            predicate = lambda index: index.get("queryable") is True
        
        while True:
            indices = list(collection.list_search_indexes('vector_index'))
            if len(indices) and predicate(indices[0]):
                break
            time.sleep(5)
        print(result + " is ready for querying.")

    def insertInforMany(self, 
                        user_name:str,
                        password: str,
                        images: np.ndarray,
                        embeddings: np.ndarray
                        ):
        """insert for checking time for one user with many images"""
        assert images.ndim == 4, f"Found {images.ndim} ndim"
        assert images.shape[0] == embeddings.shape[0], \
            f"Found {images.shape[0]} shape vs {embeddings.shape[0]} shape"

        database = self.client["Storage"]
        collection = database["Infor"]

        try:
            collection.insert_many([
                {
                    'image': img.tolist(),
                    'embedding': embedding.tolist(),
                    'user_name': user_name,
                    'password': password
                }
                for img, embedding in zip(images, embeddings)
            ])
        except Exception as e:
            print(e)
    
    def check_duplicate_name_password(self, 
                    user_name: str, 
                    password:str
                    )-> bool:
        database = self.client["Storage"]
        collection = database["Infor"]

        search_result = list(collection.find(
            {
                'user_name': user_name, 
                'password': password
            }
        ))
        if len(search_result) > 0:
            return True
        else:
            return False
    
    @staticmethod
    def _adjust_pipeline(query_embedding: np.ndarray,
                        num_all_docs:int,
                        limit_return:int
                        )->List[Dict[str,dict]]:
        """define pipeline for vector search"""
        return [
            {
                '$vectorSearch': {
                    'index': 'vector_index', 
                    'path': 'embedding',
                    'exact': True,
                    'queryVector': generate_bson_vector(query_embedding.tolist()),
                    'limit': num_all_docs
                }
            },
            {
                '$project': {
                    '_id': "$_id",
                    'user_name': "$user_name",
                    'score': {
                        '$meta': 'vectorSearchScore'
                    }
                }
            },
            {
                '$group': {
                    '_id': "$user_name",
                    'maxScore': {'$max': "$score"}
                }
            },
            {
                '$sort': {'maxScore': 1} # Sort ascending sim score
            },
            {
                '$limit': limit_return
            }
        ]

    def searchUserWithEmbeddings(self, 
                                 batch_query_embeddings: np.ndarray, 
                                 limit_return:int = 1
                                 )->str:
        """
        batch_query_embeddings: np.ndarray shape (batch, 128) or (batch, 512)
        """
        database = self.client["Storage"]
        collection = database["Infor"]
        all_docs = collection.count_documents(filter={})
        
        batch_results = {
            # run each pipeline
            ith: [{'name':item['_id'],'score':item['maxScore']} for item in
            list(collection.aggregate(Mongo_Handler._adjust_pipeline(
                                                        query_embedding= query_embedding, 
                                                        num_all_docs = all_docs,
                                                        limit_return = limit_return)
                                                )
                    )
            ][0]
            for ith, query_embedding in enumerate(batch_query_embeddings)
        }
        
        
        # ranking
        # candidate_users = {ele[0]:[] for _round in batch_results for ele in _round}
        
        # for batch in batch_results:
        #     for user, rank, _ in batch:
        #         candidate_users[user].append(rank)
        
        # # mean ranking
        # candidate_users = {k: sum(v)/limit_return for k,v in candidate_users.items()}

        candidate_names = {k:v['name'] for k,v in batch_results.items()}
        candidate_scores = {k:v['score'] for k,v in batch_results.items()}
        # # sort ranking score in ascending order
        sorted_candidate_users = {k:v for k,v in 
                                  sorted(candidate_scores.items(), key=lambda item: item[1])
                                  }
        result_index = list(sorted_candidate_users.keys())[-1]
        return candidate_names[result_index]

    def searchUserWithEmbeddings_V2(self, query_embedding: torch.Tensor = None):
        return self.UserEmbeddingSearch.search(query_embedding)
    
    # for cookie processing
    def insertCookie(self,
                     session_id: uuid.UUID,
                     action: Literal['signup','checkin']
                     )->None:
        """Insert one cookie"""
        database = self.client["Storage"]
        collection = database["Cookies"]
        try:
            collection.insert_one({'session_id': session_id, 
                                   'action': action}
                                )
        except Exception as e:
            print(e)

    
    def searchCookie(self, session_id: uuid.UUID)->List[str]:
        database = self.client["Storage"]
        collection = database["Cookies"]

        search_result = list(collection.find(
            {
                'session_id': session_id
            }
        ))

        return [item['action'] for item in search_result]
    
    