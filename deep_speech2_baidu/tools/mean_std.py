"""统计特征的 mean和 std，便于后续计算"""
import argparse
from deep_speech2_baidu.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from deep_speech2_baidu.data_utils.normalizer import FeatureNormalizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Filepath of data")
    parser.add_argument("--num_samples", type=int, default=2000, help="num of samples to stat")
    parser.add_argument("--specgram_type", type=str, choices=["linear", "mfcc"], help="Audio feature type. Option: linear, mfcc")
    parser.add_argument("--output_path", type=str, help="Filepath to write mean and std (.npz)")

    args = parser.parse_args()
    featurizer = AudioFeaturizer(specgram_type=args.specgram_type)

    def feature_func(audio_segment):
        featurizer.featurize(audio_segment, allow_upsampling=True, allow_downsampling=True)

    normalizer = FeatureNormalizer(
        mean_std_filepath=None,
        data_path=args.file_path,
        featurize_func=feature_func,
        num_samples=args.num_samples)

    normalizer.write_to_file(args.output_path)
