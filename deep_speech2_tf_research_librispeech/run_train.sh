cd /data/zhaochengming/projects/SpeechRecognition

PYTHONPATH=.:$PYTHONPATH

python deep_speech2_tf_research_librispeech/deep_speech.py \
--data_dir='/data/zhaochengming/data/data_source/Speech/LibriSpeech/total.dat' \
--model_dir='/data/zhaochengming/projects/SpeechRecognition/deep_speech2_tf_research_librispeech/logs' \
--sortagrad=True \
--sample_rate=16000 \
--window_ms=20 \
--stride_ms=10 \
--vocab_file='/data/zhaochengming/projects/SpeechRecognition/deep_speech2_tf_research_librispeech/data/vocabulary.txt' \
--rnn_hidden_size=800 \
--rnn_hidden_layers=5 \
--use_bias=True \
--is_bidirectional=True \
--rnn_type='gru' \
--learning_rate=0.0005 \
--seed=1 \
--batch_size=128 \
--train_epochs=10 \
--num_gpus=1 \
--hooks='' \
--benchmark_test_id='FirstRun'


if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0