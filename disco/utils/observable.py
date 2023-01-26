# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

class Observable(object):
    """Textbook observer pattern"""

    def __init__(self):
        self.observers = set()

    def enroll(self, function):
        self.observers.add(function)

    def dispatch(self, *args, **kwargs):
        for f in self.observers:
            f(*args, **kwargs)

def forward(observable1, observable2):
    """Forwards messages from an observable to another
    with the same signature"""
    def forwarder(*args, **kwargs):
        observable2.dispatch(*args, **kwargs)
    observable1.enroll(forwarder)

