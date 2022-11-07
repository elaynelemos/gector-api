import argparse

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel


def predict_for_file(input_file, output_file, model, batch_size=32, to_normalize=False):
    test_data = read_lines(input_file)
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

    with open(output_file, 'w') as f:
        f.write("\n".join(result_lines) + '\n')
    return cnt_corrections


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

    corrections = predict_for_sentences(sentences, model,
                                        batch_size=batch_size,
                                        to_normalize=normalize)
    # evaluate with m2 or ERRANT
    print(f"\nProduced overall corrections: {corrections}")


def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)

    cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size,
                                       to_normalize=args.normalize)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")


#### Example:

# sentences = f'Nevertheless , while I fell that a carrier should not be obligated to tell his or her relatives about the genetic risk , I would like to suggest the person to at least share the information with her or his potential partner ( husband or wife ) since the individual \'s genetic risk may affect their future children .\n'
# correct_sentences(
#     sentences=args.sentences,
#     vocab_path=args.vocab_path,
#     model_path=args.model_path,
#     max_len=args.max_len,
#     min_len=args.min_len,
#     iteration_count=args.iteration_count,
#     min_error_probability=args.min_error_probability,
#     lowercase_tokens=args.lowercase_tokens,
#     transformer_model=args.transformer_model,
#     special_tokens_fix=args.special_tokens_fix,
#     additional_confidence=args.additional_confidence,
#     batch_size=args.batch_size,
#     additional_del_confidence=args.additional_del_confidence,
#     is_ensemble=args.is_ensemble,
#     normalize=args.normalize,
#     weights=args.weights
# )


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=False)
    parser.add_argument('--sentences',
                        help='Text in case of commandline input',
                        required=False)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=False)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--normalize',
                        help='Use for text simplification.',
                        action='store_true')
    args = parser.parse_args()

    # main(args)

    correct_sentences(
        sentences=args.sentences,
        vocab_path=args.vocab_path,
        model_path=args.model_path,
        max_len=args.max_len,
        min_len=args.min_len,
        iteration_count=args.iteration_count,
        min_error_probability=args.min_error_probability,
        lowercase_tokens=args.lowercase_tokens,
        transformer_model=args.transformer_model,
        special_tokens_fix=args.special_tokens_fix,
        additional_confidence=args.additional_confidence,
        batch_size=args.batch_size,
        additional_del_confidence=args.additional_del_confidence,
        is_ensemble=args.is_ensemble,
        normalize=args.normalize,
        weights=args.weights
    )

