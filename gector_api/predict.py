from utils.helpers import normalize
from gector.gec_model import GecBERTModel


def predict_for_sentences(sentences, model, batch_size=32, to_normalize=False):
    test_data = sentences.splitlines()

    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    result_lines = [" ".join(x) for x in predictions]
    if to_normalize:
        result_lines = [normalize(line) for line in result_lines]

    return [ "\n".join(result_lines), cnt_corrections ]


def correct_sentences(sentences, vocab_path, model_path, max_len, min_len, iteration_count,
                        min_error_probability, lowercase_tokens, transformer_model,
                        special_tokens_fix, additional_confidence, batch_size,
                        additional_del_confidence, is_ensemble, normalize, weights):
    # get all paths
    model = GecBERTModel(vocab_path=vocab_path,
                         model_paths=model_path,
                         max_len=max_len, min_len=min_len,
                         iterations=iteration_count,
                         min_error_probability=min_error_probability,
                         lowercase_tokens=lowercase_tokens,
                         model_name=transformer_model,
                         special_tokens_fix=special_tokens_fix,
                         log=False,
                         confidence=additional_confidence,
                         del_confidence=additional_del_confidence,
                         is_ensemble=is_ensemble,
                         weigths=weights)

    return predict_for_sentences(sentences, model,
                                batch_size=batch_size,
                                to_normalize=normalize)
