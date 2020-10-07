#!/bin/bash

mkdir ../ADARI
mkdir ../ADARI/image_embeddings

aws s3 cp s3://mmmlf20/resnet_image_embeddings.json ../ADARI/image_embeddings/

mkdir ../ADARI/word_embeddings
aws s3 cp s3://mmmlf20/word_embeddings/fur_5c_50d_sk_glove_ft.json ../ADARI/word_embeddings/

mkdir ../ADARI/json_files
mkdir ../ADARI/json_files/ADARI_FUR_images_sentences_words/

aws s3 cp s3://mmmlf20/json_files/ADARI_fur_images_sentences_words/ADARI_v2_FUR_images_words.json ../ADARI/json_files/ADARI_FUR_images_sentences_words/
