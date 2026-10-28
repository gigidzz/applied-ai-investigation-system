"""The HTTP layer.

Deliberately thin. Every decision this package could make has a better home:
business logic lives in the domain packages, the choice of retrieval
configuration lives in `deps`, and mapping a domain error to a status code
lives in one set of exception handlers. What is left is reading a request,
calling a domain function, and shaping a response.
"""
