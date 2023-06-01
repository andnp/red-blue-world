import json

class Config:
    def __init__(self, path: str):
        self._path = path
        self._raw_data = self.load()

    def load(self):
        with open(self._path, 'r') as f:
            d = json.load(f)

        return d

    def get(self, key: str):
        if '.' not in key:
            return self._raw_data[key]

        parts = key.split('.')
        sub = self._raw_data
        for p in parts:
            sub = sub[p]

        return sub
