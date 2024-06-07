from src.utils import device
from src.raclip_modules.loader import Loader

REFERENCE_SET_FILE = './data/reference_set_10k.csv'
REFERENCE_EMBEDDINGS_FILE = './.cache/reference_embeddings_10k.pkl'

print(f'Device being used: {device}')
loader = Loader()
reference_embeddings = loader.load_reference_set(REFERENCE_SET_FILE)
loader.save_embeddings(reference_embeddings, REFERENCE_EMBEDDINGS_FILE)