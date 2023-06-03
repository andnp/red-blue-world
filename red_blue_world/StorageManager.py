
import sqlite3
import os, errno
from typing import List, Tuple, Any, Dict, AnyStr
from abc import ABCMeta, abstractmethod
import simplejson

# This class and its method are used to instantiate a storage class
# To keep things consistent, do not directly instantiate
# a storage class without going through the factory
class StoreFactory:

    @staticmethod
    def create_store(store_type: str, db_name: str = ''):
        if(store_type == "sqlite_basic"):
            return SqliteBasicStorage(db_name=db_name)
        else:
            raise NotImplementedError
        

class Store(metaclass=ABCMeta):

    @abstractmethod
    def patch_exists(self, patch_id: AnyStr) -> bool:
        pass

    @abstractmethod
    def load_patch_states(self, patch_ids: List) -> List[Tuple[Any, Any]]:
        pass

    @abstractmethod
    def load_patch_state(self, patch_id: AnyStr) -> Dict: 
        pass

    @abstractmethod
    def store_patch(self, patch_id: AnyStr, patch_state: Dict) -> None:
        pass

class SqliteBasicStorage(Store):
    _instances : Dict[Any, Any] = {}

    def __new__(cls, db_name):
        if cls not in cls._instances:
            cls._instances[cls] = super(SqliteBasicStorage, cls).__new__(cls)

        return cls._instances[cls]

    def __init__(self, db_name) -> None:
        super().__init__()
        
        # if the agent terminates and leaves behind a db, we delete that db
        try:
            os.remove(db_name)
        except OSError as e:
            if e.errno != errno.ENOENT: # an error other than that the file does not exist
                raise 

        # creates the env db
        self.con = sqlite3.connect(db_name)

        # creates the patches table
        cur = self.con.cursor()
        cur.execute("CREATE TABLE patches(patch_id TEXT PRIMARY KEY, patch_state TEXT NOT NULL)")


    def patch_exists(self, patch_id: AnyStr) -> bool:
        cur = self.con.cursor()
        exist_cmd = '''SELECT 1 FROM patches WHERE patch_id=? LIMIT 1'''
        cur.execute(exist_cmd, (patch_id,))
        exists = cur.fetchone() is not None
        return exists

    def load_patch_states(self, patch_ids: List) -> List[Tuple[Any, Any]]:
        cur = self.con.cursor()
        load_cmd = '''SELECT * FROM patches WHERE patch_id IN (%s)'''
        cur.execute(load_cmd % ','.join('?'*len(patch_ids)), patch_ids)
        ids_and_patch_states = cur.fetchall()
        json_loaded = [(patch_id, simplejson.loads(state_json)) for patch_id, state_json in ids_and_patch_states]
        return json_loaded 

    def load_patch_state(self, patch_id: AnyStr) -> Dict: 
        cur = self.con.cursor()
        load_cmd = '''SELECT patch_state FROM patches WHERE patch_id=?'''
        cur.execute(load_cmd, (patch_id,))
        str_patch_state = cur.fetchone()
        if(str_patch_state is None):
            return {}
        json_loaded = simplejson.loads(str_patch_state[0])
        return json_loaded

    def store_patch(self, patch_id: AnyStr, patch_state: Dict) -> None:
        json_rep = simplejson.dumps(patch_state)
        insert_cmd = '''INSERT INTO patches(patch_id, patch_state) VALUES(?,?) 
        ON CONFLICT(patch_id) DO UPDATE SET patch_state=excluded.patch_state'''
        cur = self.con.cursor()
        cur.execute(insert_cmd, (patch_id, json_rep))
        self.con.commit()
        

