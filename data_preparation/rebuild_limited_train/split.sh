# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
python sample_10h.py  --target_dir=10h_temp
python select_1h.py --root_10h=10h_temp --target_dir=1h
python split_1h_in10min.py --root_1h=1h --target_dir=6x10min

mkdir librispeech_release
mv 10h_temp ./librispeech_release/9h/
mv 6x10min ./librispeech_release/1h/

python clean_texts.py --root=librispeech_release/

rm -rf 1h
