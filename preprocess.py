import os
import csv
import json
import spacy
from spacy.lang.en import English
from typing import Dict, Tuple, Sequence, List, Callable

nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)

import numpy
import torch
import h5py
from pytorch_pretrained import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers.token import Token
from allennlp.common.tqdm import Tqdm


class Vectorizer:
	"""
	Abstract class for creating a tensor representation of size (#layers, #tokens, dimensionality)
	for a given sentence.
	"""
	def vectorize(self, sentence: str) -> numpy.ndarray:
		"""
		Abstract method for tokenizing a given sentence and return embeddings of those tokens.
		"""
		raise NotImplemented

	def make_hdf5_file(self, sentences: List[str], out_fn: str) -> None:
		"""
		Given a list of sentences, tokenize each one and vectorize the tokens. Write the embeddings
		to out_fn in the HDF5 file format. The index in the data corresponds to the sentence index.
		"""
		sentence_index = 0

		with h5py.File(out_fn, 'w') as fout:
			for sentence in Tqdm.tqdm(sentences):
				embeddings = self.vectorize(sentence)
				fout.create_dataset(str(sentence_index), embeddings.shape, dtype='float32', data=embeddings)
				sentence_index += 1


class ELMo(Vectorizer):
	def __init__(self):
		self.elmo = ElmoEmbedder()

	def vectorize(self, sentence: str) -> numpy.ndarray:
		"""
		Return a tensor representation of the sentence of size (3 layers, num tokens, 1024 dim).
		"""
		# tokenizer's tokens must be converted to string tokens first
		tokens = list(map(str, spacy_tokenizer(sentence)))	
		embeddings = self.elmo.embed_sentence(tokens)
		return embeddings 	


class BertBaseCased(Vectorizer):
	def __init__(self):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
		self.model = BertModel.from_pretrained('bert-base-cased')
		self.model.eval()

	def vectorize(self, sentence: str) -> numpy.ndarray:
		"""
		Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
		Even though there are only 12 layers in GPT2, we include the input embeddings as the first
		layer (for a fairer comparison to ELMo).
		"""
		# add CLS and SEP to mark the start and end
		tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']
		# tokenize sentence with custom BERT tokenizer
		token_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
		# segment ids are all the same (since it's all one sentence)
		segment_ids = numpy.zeros_like(token_ids)

		tokens_tensor = torch.tensor([token_ids])
		segments_tensor = torch.tensor([segment_ids])

		with torch.no_grad():
			embeddings, _, input_embeddings = self.model(tokens_tensor, segments_tensor)

		# exclude embeddings for CLS and SEP; then, convert to numpy
		embeddings = torch.stack([input_embeddings] + embeddings, dim=0).squeeze()[:,1:-1,:]
		embeddings = embeddings.detach().numpy()							

		return embeddings


class GPT2(Vectorizer):
	def __init__(self):
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		self.model = GPT2Model.from_pretrained('gpt2')
		self.model.eval()

	def vectorize(self, sentence: str) -> numpy.ndarray:
		"""
		Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
		Even though there are only 12 layers in GPT2, we include the input embeddings (with 
		positional information).

		For an apples-to-apples comparison with ELMo, we use the spaCy tokenizer at first -- to 
		avoid breaking up a word into subwords if possible -- and then fall back to using the GPT2 
		tokenizer if needed.
		"""
		# use spacy tokenizer at first
		tokens_tentative = list(map(str, spacy_tokenizer(sentence)))
		token_ids_tentative = self.tokenizer.convert_tokens_to_ids(tokens_tentative)

		tokens = []
		token_ids = []

		for i, tid in enumerate(token_ids_tentative):
			# if not "unknown token" ID = 0, proceed
			if tid != 0:
				tokens.append(tokens_tentative[i])
				token_ids.append(tid)
			else:
				# otherwise, try to find a non-zero token ID by preprending the special character
				special_char_tid = self.tokenizer.convert_tokens_to_ids('Ġ' + tokens_tentative[i])
				# otherwise, break up the word using the given GPT2 tokenizer
				subtoken_tids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tokens_tentative[i]))

				if special_char_tid != 0:
					tokens.append('Ġ' + tokens_tentative[i])
					token_ids.append(special_char_tid)
				else:
					tokens.extend(self.tokenizer.tokenize(tokens_tentative[i]))
					token_ids.extend(subtoken_tids)

		tokens_tensor = torch.tensor([token_ids])

		# predict hidden states features for each layer
		with torch.no_grad():
		    _, _, embeddings = self.model(tokens_tensor)

		embeddings = torch.stack(embeddings, dim=0).squeeze().detach().numpy()							
		
		return embeddings	


def index_tokens(tokens: List[str], sent_index: int, indexer: Dict[str, List[Tuple[int, int]]]) -> None:
	"""
	Given string tokens that all appear in the same sentence, append tuple (sentence index, index of
	word in sentence) to the list of values each token is mapped to in indexer. Exclude tokens that 
	are punctuation.

	Args:
		tokens: string tokens that all appear in the same sentence
		sent_index: index of sentence in the data
		indexer: map of string tokens to a list of unique tuples, one for each sentence the token 
			appears in; each tuple is of the form (sentence index, index of token in that sentence)
	"""
	for token_index, token in enumerate(tokens):
		if not nlp.vocab[token].is_punct:
			if str(token) not in indexer:
				indexer[str(token)] = []

			indexer[str(token)].append((sent_index, token_index))


def index_sentence(data_fn: str, index_fn: str, tokenize: Callable[[str], List[str]], min_count=5) -> List[str]:
	"""
	Given a data file data_fn with the format of sts.csv, index the words by sentence in the order
	they appear in data_fn. 

	Args:
		index_fn: at index_fn, create a JSON file mapping each word to a list of tuples, each 
			containing the sentence it appears in and its index in that sentence
		tokenize: a callable function that maps each sentence to a list of string tokens; identity
			and number of tokens generated can vary across functions
		min_count: tokens appearing fewer than min_count times are left out of index_fn

	Return:
		List of sentences in the order they were indexed.
	"""
	word2sent_indexer = {}
	sentences = []
	sentence_index = 0

	with open(data_fn) as csvfile:
		csvreader = csv.DictReader(csvfile, quotechar='"', delimiter='\t')

		for line in csvreader:
			# only consider scored sentence pairs
			if line['Score'] == '':	
				continue

			# handle case where \t is between incomplete quotes (causes sents to be treated as one)
			if line['Sent2'] is None:
				line['Sent1'], line['Sent2'] = line['Sent1'].split('\t')[:2]

			index_tokens(tokenize(line['Sent1']), sentence_index, word2sent_indexer)
			index_tokens(tokenize(line['Sent2']), sentence_index + 1, word2sent_indexer)
			sentences.append(line['Sent1'])
			sentences.append(line['Sent2'])
			sentence_index += 2

	# remove words that appear less than min_count times
	infrequent_words = list(filter(lambda w: len(word2sent_indexer[w]) < min_count, word2sent_indexer.keys()))
	
	for w in infrequent_words:
		del word2sent_indexer[w]

	json.dump(word2sent_indexer, open(index_fn, 'w'), indent=1)
	
	return sentences


if __name__ == "__main__":
	# where to save the contextualized embeddings
	EMBEDDINGS_PATH = "~/contextual_embeddings"

	# sts.csv has been preprocessed to remove all quotes of type ", since they are often not completed
	elmo = ELMo()
	sentences = index_sentence('sts.csv', 'elmo/word2sent.json', lambda s: list(map(str, spacy_tokenizer(s))))
	elmo.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'elmo.hdf5'))

	bert = BertBaseCased()
	sentences = index_sentence('sts.csv', 'bert/word2sent.json', bert.tokenizer.tokenize)
	bert.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'bert.hdf5'))

	gpt2 = GPT2()
	sentences = index_sentence('sts.csv', 'gpt2/word2sent.json', lambda s: list(map(str, spacy_tokenizer(s))))
	gpt2.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'gpt2.hdf5'))


			

