
cd /data/zhaochengming/projects/SpeechRecognition
PYTHONPATH=.:$PYTHONPATH

python asrt_keras/data/vocab_prepare.py \
--output='/data/zhaochengming/data/data_source/Speech/vocab.txt'

if [ $? -ne 0 ]; then
    echo "Failed in generating vocab!"
    exit 1
fi


python asrt_keras/data/data_prepare.py \
--thchs30='/data/zhaochengming/data/data_source/Speech/THCHS30/data_thchs30' \
--aishell='/data/zhaochengming/data/data_source/Speech/aishell/data_aishell' \
--output='/data/zhaochengming/data/data_source/Speech/labeled_data.txt'

if [ $? -ne 0 ]; then
    echo "Failed in generating files!"
    exit 1
fi

exit 0