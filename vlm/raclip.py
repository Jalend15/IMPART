from vlm.base_vlm import BaseVLM

class RaClip(BaseVLM):
    # Should have all necessary components including clip, reference dataset, k, model parameters etc..
    def __init__(self, name='RaClip'):
        super().__init__(name)

    # Should load and create the reference set and train set
    def load_dataset(self, dataset):
        pass

    # implement training loop
    def train(self):
        pass
        
    # should implement retrieval from reference set and augmentation  
    def encode_image(self, image_batch):
        pass

    # same as clip 
    def encode_text(self, text_batch):
        pass