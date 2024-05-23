from raclip_modules.loader import Loader

loader = Loader()
reference_embeddings = loader.load_reference_set('./data/reference_set_details.csv')
loader.save_embeddings(reference_embeddings, './data/reference_embeddings.pkl')