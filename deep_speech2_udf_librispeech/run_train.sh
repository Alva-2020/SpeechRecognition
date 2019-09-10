cd /data/zhaochengming/projects/SpeechRecognition

PYTHONPATH=.:$PYTHONPATH

python deep_speech2_udf_librispeech/main.py \
--param_file='/data/zhaochengming/projects/SpeechRecognition/deep_speech2_udf_librispeech/settings.xml' \
--data_file='/data/zhaochengming/data/data_source/Speech/LibriSpeech/total.dat' \
--vocab_file='/data/zhaochengming/projects/SpeechRecognition/deep_speech2_udf_librispeech/data/vocabulary.txt' \
--sortagrad=True \
--sample_rate=16000 \
--window_ms=20 \
--stride_ms=10 \
--is_normalize=True \
--rnn_hidden_size=800 \
--rnn_hidden_layers=5 \
--rnn_type='gru' \
--is_bidirectional=True \
--fc_use_bias=False \
--model_dir='/data/zhaochengming/projects/SpeechRecognition/deep_speech2_udf_librispeech/logs' \
--random_seed=0 \
--epochs=100 \
--batch_size=128 \
--learning_rate=0.0005 \
--gpu_num=2


if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0