# How Contextual are Contextualized Word Representations?

This is the code for "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings", presented at EMNLP 2019 (oral). See [the paper](https://www.aclweb.org/anthology/D19-1006.pdf) and [corresponding slides](https://kawine.github.io/assets/emnlp_2019_contextual_slides.pdf).

### Setup

Run the setup.sh shell script. You may also need to install the dependencies for pytorch_pretrained, which is an earlier version of the Huggingface Transformers library that has been modified to easily retrieve the contextualized representations at each layer. 

If you want to evaluate the new type of static embeddings proposed in the paper -- created by taking the first principal component of contextualized embeddings -- then you need to install dependencies for word_embeddings_benchmark, which is a fork of the [library of the same name](https://github.com/kudkudak/word-embeddings-benchmarks).

Note that preprocess.py generates a contextualized representation for every token in every sentence (x 3 models x #layers per model). These representations are stored in the HDF5 format and later accessed by analyze.py. These vectors take up a lot of space -- for the analysis done in the paper, roughly 40G of space was needed.

### Use

1. Make sure your data is in the form of STS data (see sts.csv for an example). If you do not wish to use this format, change the code in preprocess.index_sentence as needed. Note that the sample STS data included in sts.csv is a small fraction of the data used in the paper's experiments (due to size concerns).
2. Once your data is in the right format (or you have modified the code as needed), run preprocess.py. This will save the contextualized embeddings for each sentence in your data in the HDF5 file format.
3. Run analyze.py to calculate all the statistics discussed in the paper. These statistics will be saved under the bert/, elmo/, and gpt2/ subdirectories, unless you specify otherwise.
4. [optional] Create an img/ subdirectory and call the functions in visualize.py to create the charts in the paper.
5. [optional] Run visualize.evaluate to see how the new type of static embedding in the paper does on a set of standard word vector benchmarks. If you do not wish to evaluate on classical methods like GloVe and FastText, remove them from the list of vector paths. File paths in this function are hard-coded and should be modified based on where you saved the contextualized representations.

### FAQs

1.	Do the statistics in the CSV/JSON files under the bert/, elmo/, and gpt2 subdirectories match those in the paper exactly?

	No; for example, word2sent.json, which indexes words in the data, is meant to be an example and only contains one word. The statistics in the other files (e.g., self_similarity.csv) should be close to what is charted in the paper, though not an exact match. Note that the statistics in those files have not been adjusted for anisotropy, as we do in the paper; this is why, for example, the values in variance_explained.csv are so high.

### Reference

If you use this code, please cite
```
@inproceedings{ethayarajh-2019-contextual,
    title = "How Contextual are Contextualized Word Representations? Comparing the Geometry of {BERT}, {ELM}o, and {GPT}-2 Embeddings",
    author = "Ethayarajh, Kawin",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1006",
    doi = "10.18653/v1/D19-1006",
    pages = "55--65"
}
```