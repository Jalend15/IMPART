from src.raclip_modules.loader import Loader

REFERENCE_SET_FILE = './data/reference_set.csv'
REFERENCE_EMBEDDINGS_FILE = './.cache/reference_embeddings.pkl'

loader = Loader()
reference_embeddings = loader.load_reference_set(REFERENCE_SET_FILE)
loader.save_embeddings(reference_embeddings, REFERENCE_EMBEDDINGS_FILE)