#!/usr/bin/env bash
python -m spacy download en
# get pretrained Transformer ELMo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/transformer-elmo-2019.01.10.tar.gz