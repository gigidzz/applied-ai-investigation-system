"""Deciding which chunks reach the prompt.

Two retrieval routes produce two rankings that disagree, and generation can
only be given one list. This package merges them and then re-scores the
result, because merging alone cannot repair a question both routes got wrong.
"""
