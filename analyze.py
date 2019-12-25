import os
import h5py
import json
import random
import numpy as np
import itertools
import csv
from typing import Dict, Tuple, Sequence, List
from scipy.spatial.distance import cosine
from allennlp.common.tqdm import Tqdm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def calculate_word_similarity_across_sentences(
	embedding_fn: str, 
	word2sent_indexer: Dict[str, List[Tuple[int, int]]],
	out_fn : str,
	num_samples=1000) -> None:
	"""
	Each word in word2sent_indexer appears in multiple sentences. Thus each occurrence of the word 
	will have a unique embedding at each layer. For each layer, calculate the average cosine 
	similarity between embeddings of the same word across its different occurrences. This captures 
	how sentence-specific each word representation is at each layer.

	Create a table of size (#words x #layers) and write it to out_fn.
	"""
	f = h5py.File(embedding_fn, 'r')
	num_layers = f["0"].shape[0]

	# write statistics to csv file: one row per word, one column per layer
	fieldnames = ['word'] + list(map(lambda w: 'layer_' + w, map(str, range(num_layers))))
	writer = csv.DictWriter(open(out_fn, 'w'), fieldnames=fieldnames)
	writer.writeheader()

	for word in Tqdm.tqdm(word2sent_indexer):
		similarity_by_layer = { 'word' : word }

		# list of tuples (sentence index, index of word in sentence) for word occurrences
		occurrences = word2sent_indexer[word]

		indices = range(len(occurrences))
		index_pairs = [ (i,j) for i,j in itertools.product(indices, indices) if i != j ]
	
		# for very frequent words (e.g., stopwords), there are too many pairwise comparisons
		# so for faster estimation, take a random sample of no more than num_samples pairs		
		if len(index_pairs) > num_samples:
			index_pairs = random.sample(index_pairs, num_samples)

		# calculate statistic for each layer using sampled data
		for layer in range(num_layers):
			layer_similarities = []

			for i,j in index_pairs:
				sent_index_i, word_in_sent_index_i = occurrences[i]
				embedding_i = f[str(sent_index_i)][layer, word_in_sent_index_i]

				sent_index_j, word_in_sent_index_j = occurrences[j]
				embedding_j = f[str(sent_index_j)][layer, word_in_sent_index_j]

				layer_similarities.append(1 - cosine(embedding_i, embedding_j))

			mean_layer_similarity = round(np.nanmean(layer_similarities), 3)
			similarity_by_layer[f'layer_{layer}'] = mean_layer_similarity

		writer.writerow(similarity_by_layer)


def variance_explained_by_pc(
	embedding_fn: str, 
	word2sent_indexer: Dict[str, List[Tuple[int, int]]],
	variance_explained_fn : str,
	pc_fn : str) -> None:
	"""
	Each word in word2sent_indexer appears in multiple sentences. Thus each occurrence of the word 
	will have a different embedding at each layer. How much of the variance in these occurrence 
	embeddings can be explained by the first principal component? In other words, to what extent
	can these different occurrence embeddings be replaced by a single, static word embedding?
	
	Create a table of size (#words x #layers) and write the variance explained to variance_explained_fn.
	Write the first principal component for each word to pc_fn + str(layer_index), where each row 
	starts with a word followed by space-separated numbers.
	"""
	f = h5py.File(embedding_fn, 'r')
	num_layers = f["0"].shape[0]

	# write statistics to csv file: one row per word, one column per layer
	# excluding first layer, since we don't expect the input embeddings to be the same at all for gpt2/bert
	# and we expect them to be identical for elmo
	fieldnames = ['word'] + list(map(lambda w: 'layer_' + w, map(str, range(1, num_layers))))
	writer = csv.DictWriter(open(variance_explained_fn, 'w'), fieldnames=fieldnames)
	writer.writeheader()

	# files to write the principal components to 
	pc_vector_files = { layer: open(pc_fn + str(layer), 'w') for layer in range(1, num_layers) }

	for word in Tqdm.tqdm(word2sent_indexer):
		variance_explained = { 'word' : word }

		# calculate variance explained by the first principal component
		for layer in range(1, num_layers):
			embeddings = [ f[str(sent_index)][layer, word_index].tolist() for sent_index, word_index 
				in word2sent_indexer[word] if f[str(sent_index)][layer, word_index].shape != () ]

			pca = PCA(n_components=1)
			pca.fit(embeddings)
			
			variance_explained[f'layer_{layer}'] = min(1.0, round(pca.explained_variance_ratio_[0], 3))
			pc_vector_files[layer].write(' '.join([word] + list(map(str, pca.components_[0]))) + '\n')

		writer.writerow(variance_explained)

		
def explore_embedding_space(
	embedding_fn: str, 
	out_fn : str,
	num_samples=1000) -> None:
	"""
	Calculate the following statistics for each layer of the model:
	1. mean cosine similarity between a sentence and its words
	2. mean dot product between a sentence and its words
	3. mean word embedding norm
	4. mean cosine similarity between randomly sampled words
	5. mean dot product between randomly sampled words
	6. mean variance explained by first principal component for a random sample of words

	num_samples sentences/words are used to estimate each of these metrics. We randomly sample words
	by first uniformly randomly sampling sentences and then uniformly randomly sampling a single word
	from each sampled sentence. This is because:
		- 	When we say we are interested in the similarity between random words, what we really 
			mean is the similarity between random _word occurrences_ (since each word has a unique 
			vector based on its context).
		- 	By explicitly sampling from different contexts, we avoid running into cases where two
			words are similar due to sharing the same context.

	Create a dictionary mapping each layer to a dictionary of the statistics write it to out_fn.
	"""
	f = h5py.File(embedding_fn, 'r')
	num_layers = f["0"].shape[0]
	num_sentences = len(f)

	sentence_indices = random.sample(list(range(num_sentences)), num_samples)

	mean_cos_sim_between_sent_and_words = { f'layer_{layer}' : [] for layer in range(num_layers) }
	mean_cos_sim_across_words = { f'layer_{layer}' : -1 for layer in range(num_layers) }
	word_norm_std = { f'layer_{layer}' : -1 for layer in range(num_layers) }
	word_norm_mean = { f'layer_{layer}' : -1 for layer in range(num_layers) }
	variance_explained_random = { f'layer_{layer}' : -1 for layer in range(num_layers) }

	for layer in Tqdm.tqdm(range(num_layers)):
		word_vectors = []
		word_norms = []
		mean_cos_sims = []
		mean_dot_products = []

		for sent_index in sentence_indices:
			# average word vectors to get sentence vector
			sentence_vector = f[str(sent_index)][layer].mean(axis=0)
			num_words = f[str(sent_index)].shape[1]

			# randomly add a word vector (not all of them, because that would bias towards longer sentences)
			word_vectors.append(f[str(sent_index)][layer, random.choice(list(range(num_words)))])

			# what is the mean cosine similarity between the sentence and its words?
			mean_cos_sim = np.nanmean([ 1 - cosine(f[str(sent_index)][layer,i], sentence_vector) 
				for i in range(num_words) if f[str(sent_index)][layer, i].shape != () ])
			mean_cos_sims.append(round(mean_cos_sim, 3))

			# what is the mean embedding norm across words?
			word_norms.extend([np.linalg.norm(f[str(sent_index)][layer,i]) for i in range(num_words)])

		mean_cos_sim_between_sent_and_words[f'layer_{layer}'] = round(float(np.mean(mean_cos_sims)), 3)
		mean_cos_sim_across_words[f'layer_{layer}'] = round(np.nanmean([ 1 - cosine(random.choice(word_vectors), 
			random.choice(word_vectors)) for _ in range(num_samples)]), 3)
		word_norm_std[f'layer_{layer}'] = round(float(np.std(word_norms)), 3)
		word_norm_mean[f'layer_{layer}'] = round(float(np.mean(word_norms)), 3)

		# how much of the variance in randomly chosen words can be explained by their first principal component?
		pca = TruncatedSVD(n_components=100)
		pca.fit(word_vectors)
		variance_explained_random[f'layer_{layer}'] = min(1.0, round(float(pca.explained_variance_ratio_[0]), 3))

	json.dump({
		'mean cosine similarity between sentence and words' : mean_cos_sim_between_sent_and_words,
		'mean cosine similarity across words' : mean_cos_sim_across_words,
		'word norm std' : word_norm_std,
		'word norm mean' : word_norm_mean,
		'variance explained for random words' : variance_explained_random
		}, open(out_fn, 'w'), indent=1)


if __name__ == "__main__":
	# where the contextualized embeddings are saved (in HDF5 format)
	EMBEDDINGS_PATH = "~/contextual_embeddings"

	for model in ["elmo", "bert", "gpt2"]:
		print(f"Analyzing {model} ...")

		word2sent_indexer = json.load(open(f'{model}/word2sent.json', 'r'))
		scores = json.load(open(f'{model}/scores.json', 'r'))
		EMBEDDINGS_FULL_PATH = os.path.join(EMBEDDINGS_PATH, f'{model}.hdf5')

		print(f"Analyzing word similarity across sentences ...")
		calculate_word_similarity_across_sentences(EMBEDDINGS_FULL_PATH, word2sent_indexer, 
			f'{model}/self_similarity.csv')

		print(f"Analyzing variance explained by first principal component ...")
		variance_explained_by_pc(EMBEDDINGS_FULL_PATH, word2sent_indexer,
			f'{model}/variance_explained.csv', os.path.join(EMBEDDINGS_PATH, f'pcs/{model}.pc.'))

		print(f"Exploring embedding space ...")
		explore_embedding_space(EMBEDDINGS_FULL_PATH, f'{model}/embedding_space_stats.json')

	
