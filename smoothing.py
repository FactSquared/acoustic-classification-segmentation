import numpy as np
from feature import FRAME_SIZE


def smooth(predictions, threshold=0, binary=False):
    if threshold > 0:
        predictions = merge_short_sounds(predictions, threshold)
    grouped = group_frames(predictions, binary)
    return grouped


def merge_short_sounds(predictions, threshold):  # TODO: Doesn't catch short start or end (see jazz eg)
    if len(predictions) * FRAME_SIZE < threshold:  # if audio is shorter than the threshold, do nothing
        return predictions

    merged = predictions
    i = 0
    while i < len(predictions):
        current = merged[i]
        next_different = np.where(predictions[i:] != current)[0]
        if len(next_different) == 0:  # if no more changes
            seg_length = len(predictions)-1 - i
            i += 1  # prevent infinite loop
        else:
            seg_length = next_different[0]
        if seg_length * FRAME_SIZE < threshold:
            merged[i:i+seg_length] = merged[i-1]  # replace short seg with previous value
        i += seg_length
    return merged


def group_frames(predictions, binary):
    i = 0
    current = predictions[0]
    results = []
    while i < len(predictions):
        next_different = np.where(predictions[i:] != current)[0]
        if len(next_different) == 0:  # no more changes in type left.
            results.append({
                "label": num_to_label(current, binary),
                "start": i * FRAME_SIZE/1000,
                "end": (len(predictions)-1) * FRAME_SIZE/1000
            })
            break
        seg_length = int(next_different[0])
        results.append({
            "label": num_to_label(current, binary),
            "start": i* FRAME_SIZE/1000,
            "end": (i+seg_length-1) * FRAME_SIZE/1000
        })
        i += seg_length
        current = predictions[i]
    return results


def num_to_label(i, binary):
    from feature import labels
    keys = list(labels.keys())
    vals = list(labels.values())
    if binary:
        if i > 0:
            return f"non-{keys[0]}"
        return keys[0]
    return keys[vals.index(i)]
