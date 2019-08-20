cd /data/zhaochengming/projects/SpeechRecognition

PYTHONPATH=.:$PYTHONPATH
python deep_speech2_librispeech/data/prepare.py \
--source_path='/data/zhaochengming/data/data_source/Speech/LibriSpeech'
--output_path='/data/zhaochengming/data/data_source/Speech/librispeech.txt'

if [ $? -ne 0 ]; then
    echo "Prepare data failed. Terminated."
    exit 1
fi


python deep_speech2/data/prepare.py \
--thchs30='/data/zhaochengming/data/data_source/Speech/THCHS30/data_thchs30' \
--aishell='/data/zhaochengming/data/data_source/Speech/aishell/data_aishell' \
--output='/data/zhaochengming/data/data_source/Speech/labeled_data.txt'


if [ $? -ne 0 ]; then
    echo "Prepare Labeled data failed. Terminated."
    exit 1
fi

python deep_speech2/tools/mean_std.py \
--file_path='/data/zhaochengming/data/data_source/Speech/labeled_data.txt' \
--num_samples=5000 \
--specgram_type='linear' \
--output_path='/data/zhaochengming/data/data_source/Speech/mean_std.npz'


if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


echo "Data preparation done."
exit 0