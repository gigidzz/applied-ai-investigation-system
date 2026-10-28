"""Turning retrieved chunks into a grounded, cited answer.

Everything here is built on one rule: the model may use only the text it is
given. That constraint is what makes citations checkable, abstention possible,
and contradiction reporting meaningful — a model answering from its own
parameters cannot tell the difference between knowing and guessing.
"""
