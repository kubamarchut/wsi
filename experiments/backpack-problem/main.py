#!/usr/bin/env python

from data_loader import load_data
from tests import get_all_possabilities


def main():
    data = load_data("data.csv")
    print(data)
    get_all_possabilities(data)


if __name__ == "__main__":
    main()
