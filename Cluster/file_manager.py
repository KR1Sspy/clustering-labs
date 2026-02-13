import csv
import pandas as pd


def read_csv_file(filename):
    data = pd.read_csv(filename, encoding='latin-1')
    return data.values, data.columns.values


def write_csv_file(filename, d):
    pass