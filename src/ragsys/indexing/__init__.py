"""Turning chunks into something searchable.

Two representations of the same chunk are built here, because they fail in
opposite directions. A dense embedding captures meaning and handles
paraphrase, but treats `pay-gw-4.2.1` and `pay-gw-4.2.0` as near-identical. A
sparse term vector captures exact wording and separates those two cleanly, but
cannot connect "card authorisation failing" to "payment gateway returned 503".

Retrieval runs both and merges them, so this stage must produce both.
"""
