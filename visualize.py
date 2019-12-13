import os
import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from word_embeddings_benchmark.web.evaluate import evaluate_on_all
from word_embeddings_benchmark.web.embeddings import fetch_GloVe, load_embedding

matplotlib.rc('axes', edgecolor='k')


def visualize_embedding_space():
	"""Plot the baseline charts in the paper. Images are written to the img/ subfolder."""
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:']

	for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
		x = np.array(range(num_layers))
		data = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
		plt.plot(x, [ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ], icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)
		print(spearmanr(
			[ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ],
			[ data["word norm std"][f'layer_{i}'] for i in x ]
		))

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper left')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(0,1.0)
	plt.title("Average Cosine Similarity between Randomly Sampled Words")
	plt.savefig(f'img/mean_cosine_similarity_across_words.png', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:']

	for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
		x = np.array(range(num_layers))
		data = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
		y1 = np.array([ data["mean cosine similarity between sentence and words"][f'layer_{i}'] for i in x ])
		y2 = np.array([ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ])
		plt.plot(x, y1 - y2, icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper right')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(-0.1, 0.5)
	plt.title("Average Intra-Sentence Similarity (anisotropy-adjusted)")
	plt.savefig(f'img/mean_cosine_similarity_between_sentence_and_words.png', bbox_inches='tight')
	plt.close()


def visualize_self_similarity():
	"""Plot charts relating to self-similarity. Images are written to the img/ subfolder."""
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:']

	# plot the mean self-similarity but adjust by subtracting the avg similarity between random pairs of words
	for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
		embedding_stats = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
		self_similarity = pd.read_csv(f'{model}/self_similarity.csv')

		x = np.array(range(num_layers))
		y1 = np.array([ self_similarity[f'layer_{i}'].mean() for i in x ])
		y2 = np.array([ embedding_stats["mean cosine similarity across words"][f'layer_{i}'] for i in x ])
		plt.plot(x, y1 - y2, icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper right')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(0,1)
	plt.title("Average Self-Similarity (anisotropy-adjusted)")
	plt.savefig(f'img/self_similarity_above_expected.png', bbox_inches='tight')
	plt.close()

	# list the top 10 words that are most self-similar and least self-similar 
	most_self_similar = []
	least_self_similar = []
	models = []

	for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
		self_similarity = pd.read_csv(f'{model}/self_similarity.csv')
		self_similarity['avg'] = self_similarity.mean(axis=1)

		models.append(model)
		most_self_similar.append(self_similarity.nlargest(10, 'avg')['word'].tolist())
		least_self_similar.append(self_similarity.nsmallest(10, 'avg')['word'].tolist())
	
	print(' & '.join(models) + '\\\\')
	for tup in zip(*most_self_similar): print(' & '.join(tup) + '\\\\')
	print()
	print(' & '.join(models) + '\\\\')
	for tup in zip(*least_self_similar): print(' & '.join(tup) + '\\\\')


def visualize_variance_explained():
	"""Plot chart for variance explained. Images are written to the img/ subfolder."""
	bar_width = 0.2
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:']

	# plot the mean variance explained by first PC for occurrences of the same word in different sentences
	# adjust the values by subtracting the variance explained for random sentence vectors
	for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
		embedding_stats = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
		data = pd.read_csv(f'{model}/variance_explained.csv')

		x = np.array(range(1, num_layers))
		y1 = np.array([ data[f'layer_{i}'].mean() for i in x ])
		y2 = np.array([ embedding_stats["variance explained for random words"][f'layer_{i}'] for i in x])
		plt.bar(x + i * bar_width, y1 - y2, bar_width, label=model, color=icons[i][0], alpha=0.65)

	plt.grid(True, linewidth=0.25, axis='y')
	plt.legend(loc='upper right')
	plt.xlabel('Layer')
	plt.xticks(x + i * bar_width / 2, x)
	plt.ylim(0,0.1)
	plt.axhline(y=0.05, linewidth=1, color='k', linestyle='--')
	plt.title("Average Maximum Explainable Variance (anisotropy-adjusted)")
	plt.savefig(f'img/variance_explained.png', bbox_inches='tight')
	plt.show()
	plt.close()


def evaluate():
	"""
	Evaluate both typical word embeddings (GloVe, FastText) and word embeddings created by taking the
	first PC of contextualized embeddings on standard benchmarks for word vectors (see paper for 
	details). These benchmarks include tasks like arithmetic analogy-solving. Paths in this function 
	are hard-coded and should be modified as needed.

	First, create a smaller version of the embedding files where vocabulary contains only the words
	that have some ELMo vector. Then, run the code in the word_embeddings_benchmarks library on the
	trimmed word vector files.

	Returns:
		A DataFrame where the index is the model layer from which the PC vectors are derived and
		each column contains the performance on a particular task. See the word_embeddings_benchmarks
		library for details on these tasks.
	"""
	# create a smaller version of the embedding files where vocabulary = only words that have some ELMo vector
	# vocabulary needs to be the same across all embeddings for apples-to-apples comparison
	words = set([ w.lower() for w in json.load(open('elmo/word2sent.json')).keys() ])

	# paths to GloVe and FastText word vectors
	vector_paths = [
		"~/csPMI/glove.42B.300d.txt",
		"~/csPMI/wiki.en.vec"
	]

	# paths to the principal components of the contextualized embeddings
	# each layer of each model (ELMo, BERT, GPT2) should have its own set
	pc_path = "~/contextual_embeddings/pcs"

	# paths to ELMo embeddings
	for i in range(1,3):
		vector_paths.append(os.path.join(pc_path, f'elmo.pc.{i}'))

	# paths to BERT and GPT2 embeddings
	for i in range(1,13):
		vector_paths.append(os.path.join(pc_path, f'bert.pc.{i}'))
		vector_paths.append(os.path.join(pc_path, f'gpt2.pc.{i}'))

	# where to put the smaller embedding files
	trimmed_embedding_path = "~/contextual_embeddings/trimmed/"

	for path in tqdm(vector_paths):
		name = path.split('/')[-1]

		with open(os.path.join(trimmed_embedding_path, name), 'w') as f_out:
			for line in open(path):
				if line.split()[0].lower() in words:
					f_out.write(line.strip() + '\n')

	results = []
	# run the word_embedding_benchmarks code on the trimmed word vector files
	for fn in tqdm(os.listdir(trimmed_embedding_path)):
		pth = os.path.join(trimmed_embedding_path, fn)
		
		load_kwargs = {}
		load_kwargs['vocab_size'] = sum(1 for line in open(pth))
		load_kwargs['dim'] = len(next(open(pth)).split()) - 1

		embeddings = load_embedding(pth, format='glove', normalize=True, lower=True, 
			clean_words=False, load_kwargs=load_kwargs)
		df = evaluate_on_all(embeddings)
		df['Model'] = fn
		results.append(df)

	results = pd.concat(results).set_index('Model')
	return results


