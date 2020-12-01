import collections
import random


class TrainingInstance(object):
    """A single training instance (sentence pair)."""
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels


rng = random.Random()
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, prob_mask, vocab_words):

    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for i, token in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = int(round(len(tokens) * prob_mask))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        elif rng.random() < 0.5:
            masked_token = tokens[index]
        else:
            masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    # # pad to same length
    # masked_lm_positions = (masked_lm_positions + [-1 for _ in output_tokens])[:len(output_tokens)]
    # masked_lm_labels = (masked_lm_labels + ['pad' for _ in output_tokens])[:len(output_tokens)]

    return output_tokens, masked_lm_positions, masked_lm_labels


if __name__ == '__main__':
    test_input = ['i', 'love', 'you', 'do', 'you', 'like', 'me']
    test_vocab = ['i', 'love', 'you', 'do', 'like', 'me']
    out1, out2, out3 = create_masked_lm_predictions(test_input, 0.5, test_vocab)
