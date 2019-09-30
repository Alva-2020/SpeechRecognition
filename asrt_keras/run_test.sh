
cd /data/zhaochengming/projects/SpeechRecognition
PYTHONPATH=.:$PYTHONPATH

python asrt_keras/test.py \
--data_file='/data/zhaochengming/data/data_source/Speech/labeled_data.txt' \
--vocab_file='/data/zhaochengming/data/data_source/Speech/vocab.txt' \
--model_dir='/data/zhaochengming/projects/SpeechRecognition/asrt_keras/model_log' \
--sortagrad=True \
--sample_rate=16000 \
--window_ms=20 \
--stride_ms=10 \
--is_normalize=True \
--seed=1 \
--batch_size=8 \
--gpu_num=1


if [ $? -ne 0 ]; then
    echo "Failed in testing!"
    exit 1
fi