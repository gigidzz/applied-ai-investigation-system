"""Measuring whether the system works.

Retrieval and answering are measured separately and never averaged together,
because they fail independently: a question can be answered correctly from
documents the answer key does not list, and the right documents can be
retrieved and then mishandled. One blended score would hide both.
"""
