class BaseVLM:
    def __init__(self, name):
        self.name = name
    
    def get_preprocess_transform(self):
        pass
    
    def encode_text(self, text_batch):
        pass

    def encode_image(self, image_batch):
        pass