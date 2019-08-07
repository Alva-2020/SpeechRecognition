cd /data/zhaochengming/projects/SpeechRecognition

PYTHONPATH=.:$PYTHONPATH

python deep_speech2/main.py \
--param_file='/data/zhaochengming/projects/SpeechRecognition/deep_speech2/settings.xml' \
--data_file='/data/zhaochengming/data/data_source/Speech/labeled_data.txt' \
--vocab_file='/data/zhaochengming/data/data_source/Speech/vocab.txt' \
--vocab_type='pny' \
--mean_std_file='/data/zhaochengming/data/data_source/Speech/mean_std.npz' \
--sample_rate=16000 \
--window_ms=20 \
--stride_ms=10 \
--specgram_type='linear' \
--use_dB_normalization=True \
--rnn_hidden_size=800 \
--rnn_hidden_layers=5 \
--rnn_type='gru' \
--fc_use_bias=False \
--model_dir='/data/zhaochengming/projects/SpeechRecognition/deep_speech2/AM_Model_LOG' \
--random_seed=0 \
--epochs=100 \
--batch_size=4 \
--learning_rate=0.0005


if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0