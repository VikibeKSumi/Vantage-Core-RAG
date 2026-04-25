from indicnlp.normalize.indic_normalize import DevanagariNormalizer

class TextUtils():

    def __init__(self):
        self.normalizer = DevanagariNormalizer()

    def normalize(self, query: str) -> str:
        return self.normalizer.normalize(query)