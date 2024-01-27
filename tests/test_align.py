from pii_data_detection.align import LabelTokenAligner


def test_align_labels_with_tokens():
    label_token_aligner = LabelTokenAligner()
    labels = [3, 0, 7, 0, 0, 0, 7, 0, 0]
    word_ids = [None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
    assert label_token_aligner(labels, word_ids) == [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]