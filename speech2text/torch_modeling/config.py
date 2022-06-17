from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor


SAMPLING_RATE = 16000

TOKENIZER = Wav2Vec2CTCTokenizer(
    "outputs/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)


PROCESSOR = Wav2Vec2Processor(feature_extractor=FEATURE_EXTRACTOR, tokenizer=TOKENIZER)
