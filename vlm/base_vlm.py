class BaseVLM:
    def __init__(self, name):
        self.name = name
        self.preprocess = None
    
    def encode_text(self, text_batch):
        pass

    def encode_image(self, image_batch):
        pass