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
                    self._ini_insert(init_data= init_data)
                    vector_dim = int(master_config['vector_emebd_dim'])
                    self._make_search_index(vector_dim = vector_dim)

    def close(self):
        self.client.close()

    def _ini_insert(self, 
                    init_data: List[Dict[str,Union[str,np.ndarray]]]):
        """
        Run for first time setup
        [
            {
                'image': img dtype: np.ndarray,
                'embedding': embedding dtype: np.ndarray,
                'user_name': some_name
            }
        ]
        """
        database = self.client["Storage"]
        try:
            collection = database.create_collection(name= 'Infor')
            for i in range(len(init_data)):
                init_data[i]['embedding'] = generate_bson_vector(init_data[i]['embedding'])
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
                    "similarity": "dotProduct"
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
                                 limit_return:int = 3
                                 )->str:
        """
        batch_query_embeddings: np.ndarray shape (batch, 128) or (batch, 512)
        """
        database = self.client["Storage"]
        collection = database["Infor"]
        all_docs = collection.count_documents(filter={})
        
        batch_results = [
            # run each pipeline
            [(item['_id'], ith+1, item['maxScore']) for ith, item in
            enumerate(list(collection.aggregate(Mongo_Handler._adjust_pipeline(
                                                        query_embedding= query_embedding, 
                                                        num_all_docs = all_docs,
                                                        limit_return = limit_return)
                                                )
                    ))
            ]
            for query_embedding in batch_query_embeddings
        ]
        
        # print('batch_results: ',batch_results)
        # ranking
        candidate_users = {ele[0]:[] for round in batch_results for ele in round}
        
        for batch in batch_results:
            for user, rank,_ in batch:
                candidate_users[user].append(rank)
        
        # mean ranking
        candidate_users = {k: sum(v)/limit_return for k,v in candidate_users.items()}

        # sort ranking score in ascending order
        sorted_candidate_users = {k:v for k,v in 
                                  sorted(candidate_users.items(), key=lambda item: item[1])
                                  }
        
        return list(sorted_candidate_users.keys())[-1]
    
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
    
    