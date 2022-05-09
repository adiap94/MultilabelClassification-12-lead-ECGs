#!/usr/bin/env python

import os, sys
from train_12ECG_classifier import train_12ECG_classifier

def main():

    print('Running training code...')

    train_12ECG_classifier()

    print('Done.')
if __name__ == '__main__':
    main()
    pass