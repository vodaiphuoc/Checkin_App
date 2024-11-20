import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db
import json
import numpy as np
import cv2 as cv
from typing import Union, Tuple, Literal
import uuid


class Firebase_Base(object):
    def __init__(self, master_cfg: dict) -> None:
        self.cfg = master_cfg
        self.app = self.init_app()
        assert self.app != False

    def init_app(self)->firebase_admin.App:
        cred = credentials.Certificate(self.cfg['account_config'])
        default_app = firebase_admin.initialize_app(cred, {
            'databaseURL': self.cfg['dbURL'],
            'storageBucket': self.cfg['storageBucketURL'],
            'projectID': self.cfg['project_id']
            })

        if isinstance(default_app, firebase_admin.App):
            return default_app
        else:
            return False

class Firebase_Handler(Firebase_Base):
    def __init__(self,master_cfg:dict) -> None:
        super().__init__(master_cfg)
    
    def _pushImage2Storage(self, 
                           destimation_blob_name:str,
                           source_image: Union[str,np.ndarray]
                           )->None:
        
        bucket = storage.bucket(name=self.cfg['storageBucketURL'], app= self.app)
        blob = bucket.blob('Images/'+destimation_blob_name)
        
        if isinstance(source_image,str):
            source_image = cv.imread(source_image)
        
        # convert numpy array to string of bytes
        blob.upload_from_string(source_image.tobytes())
        return None

    def _pushEmbedding2Storage(self, 
                               destimation_blob_name:str,
                               embedding: np.ndarray
                               )->None:
        
        bucket = storage.bucket(name=self.cfg['storageBucketURL'], app= self.app)
        blob = bucket.blob('Embeddings/'+destimation_blob_name)
        blob.upload_from_string(embedding.tobytes())
        return None
    
    def _pushInfor2Database(self, 
                            user_name: str, 
                            blob_name:str, 
                            password:str
                            )->None:
        ref = db.reference('Master', app= self.app, url= self.cfg['dbURL'])
        ref.push().set({'user_name': user_name, 
                        'blob_name': blob_name, 
                        'password': password})

    def insert(self,
              user_name:str, 
              image: Union[str, np.ndarray] = None, 
              embedding: np.ndarray = None,
              password: str = '123'
              ):
        blob_name = user_name+'/'+str(uuid.uuid1())
        
        self._pushImage2Storage(destimation_blob_name = blob_name, 
                                source_image= image)
        self._pushEmbedding2Storage(destimation_blob_name = blob_name,
                                    embedding= embedding)
        self._pushInfor2Database(user_name,blob_name, password)
        return None

    def check_duplicate_name_password(self, 
                            username:str, 
                            password:str
                            ):
        ref = db.reference('Master', app= self.app, url= self.cfg['dbURL'])
        search_result =  [k for k,v in ref.get().items() 
                            if v['username'] == username and v['password'] == password]
        if len(search_result) > 0:
            return True
        else:
            return False

    # for cookie processing
    def insertCookie(self, 
                     session_id: uuid.UUID,
                     action: Literal['signup','checkin']
                     )->None:
        ref = db.reference('Cookie', app= self.app, url= self.cfg['dbURL'])
        ref.push().set({'session_id': session_id, 'action': action})
        return None
    
    def searchCookie(self, session_id: uuid.UUID):
        ref = db.reference('Cookie', app= self.app, url= self.cfg['dbURL'])
        return [v['action'] for _,v in ref.get().items() if v['session_id'] == session_id]
    
    # for embedding loading
    def get_dataset(self):
        return Firebase_Dataset(cfg= self.cfg, app= self.app)


class Firebase_Dataset(object):
    def __init__(self, cfg: dict, app: firebase_admin.App) -> None:
        self.cfg = cfg
        self.app = app

        self.bucket = storage.bucket(name=self.cfg['storageBucketURL'],
                                     app= self.app)
        ref = db.reference('Master', 
                           app= self.app, 
                           url= self.cfg['dbURL'])

        self.all_database = [v for _,v in ref.get().items()] # List[Dict[str,Any]]

    def __len__(self):
        return len(self.all_database)

    def __getitem__(self, index:int) ->Tuple[str, np.ndarray]:
        """
        Decode embedding byte to numpy array dtype float32
        """
        current_dict = self.all_database[index]
        blob = self.bucket.blob('Embeddings/'+current_dict['blob_name'])
        embedding_byte = blob.download_as_string()
        embedding = np.frombuffer(embedding_byte, dtype= np.float32)
        return current_dict['user_name'], embedding
