import numpy as np

def gamma_h2(T):

    if T < 300:
        return 1.41

    if T < 1000:
        return 1.39

    if T < 2000:
        return 1.35

    if T < 4000:
        return 1.30

    return 1.28


def cp_h2(T):

    if T < 500:
        return 14300

    if T < 1500:
        return 15000

    if T < 3000:
        return 16500

    return 17500


def cv_h2(T):

    return cp_h2(T) / gamma_h2(T)