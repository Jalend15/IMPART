from vlm.base_vlm import BaseVLM

class RaClipVanilla(BaseVLM):
    # Should have all necessary components including clip, reference dataset, k etc..
    def __init__(self, name='RaClip Vanilla'):
        super().__init__(name)

    # Should load and create the reference set
    def load_reference_set(self, dataset):
        pass

    # should implement retrieval from reference set and augmentation  
    def encode_image(self, image_batch):
        pass

    # same as clip 
    def encode_text(self, text_batch):
        pass