#!/usr/bin/env bash
python ATN_Guide.py --train_file_path='data/restaurant/train.txt' --test_file_path='data/restaurant/test.txt' --embedding_file_path='data/restaurant/restaurant_2014_840b_300.txt' --saver_checkpoint='data/restaurant/checkpoint' --pre_trained_path='data/yelp/checkpoint/'
python ATN_Guide.py --train_file_path='data/laptop/train.txt' --test_file_path='data/laptop/test.txt' --embedding_file_path='data/laptop/laptop_2014_840b_300.txt' --saver_checkpoint=='data/laptop/checkpoint' --pre_trained_path='data/amazon/checkpoint8/'
