
echo "DeepSpeech2 processing..."
cd /data/zhaochengming/projects/SpeechRecognition
PYTHONPATH=.:$PYTHONPATH

python deep_speech2_librispeech/data/prepare.py \
--source_path='/data/zhaochengming/data/data_source/Speech/LibriSpeech/data' \
--output_path='/data/zhaochengming/data/data_source/Speech/LibriSpeech/total.dat'

if [ $? -ne 0 ]; then
    echo "Failed in data generating!"
    exit 1
fi

exit 0