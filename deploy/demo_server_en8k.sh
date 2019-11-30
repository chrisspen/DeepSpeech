#!/bin/bash
#--num_proc_bsearch=5
#--infer_manifest='data/librispeech/manifest.test-clean'
#--error_rate_type='wer'
python deploy/demo_server.py \
    --beam_size=500 \
    --num_conv_layers=2 \
    --num_rnn_layers=3 \
    --rnn_layer_size=1024 \
    --alpha=1.4 \
    --beta=0.35 \
    --cutoff_prob=1.0 \
    --cutoff_top_n=40 \
    --use_gru=False \
    --use_gpu=False \
    --share_rnn_weights=False \
    --mean_std_path='models/baidu_en8k/mean_std.npz' \
    --vocab_path='models/baidu_en8k/vocab.txt' \
    --model_path='models/baidu_en8k' \
    --lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm' \
    --decoding_method='ctc_beam_search' \
    --specgram_type='linear'
