"""Some constant param settings"""


class MODELKEYS:
    """The names for keras model's `Input` layer, which are at the same time the feed data's mapping keys.
    This setting can help coordinating model's setting and data generator's output definition and avoiding hard-coding.
    """
    INPUT = "the_inputs"
    INPUT_LENGTH = "the_input_length"
    LABELS = "the_labels"
    LABEL_LENGTH = "the_label_length"
