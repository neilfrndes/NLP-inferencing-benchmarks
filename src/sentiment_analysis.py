
from typing import List
import argparse
import logging
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Generates a model pickle.')
parser.add_argument(
    '-i', '--inputfile', type=str, help="Name of the input file")
parser.add_argument(
    '-o', '--outputfile', type=str, default="../models/sentiment-analysis.torch", 
    help="Name of the output file")

args = parser.parse_args()

def get_sentence(line: str, column_number: int) -> str:
    """
    Parse the Tab Separated Line to return 
    the sentence coloumn.
    """
    sentence_list = line.split('\t')  # Tab separation
    sentence_column = sentence_list[1]  # Second Column
    sentence = sentence_column[:-1]  # Remove newline
    return sentence


def parse_input(file_name: str, column_number: int) -> List[str]:
    """
    Read a specific column of the input data file.
    This function can be replaced by pandas.read_csv or the csv library
    """
    with open(file_name) as f:
        raw_data = f.readlines()
        raw_data = raw_data[1:]  # Skip header
        logger.info(f"Read {len(raw_data)} lines.")

    # Extract column
    data = [get_sentence(line, column_number) for line in raw_data]
    return data


def main(file_name: str) -> None:
    logger.info(f"Reading {file_name}...")

    logger.info("Initializing sentiment analysis model ...")
    nlp = pipeline('sentiment-analysis')
    nlp.data = parse_input(file_name, column_number=1)
    
    logger.info("Writing trained model to sentiment-analysis.model")
    torch.save(nlp, args.outputfile)
    
    logger.info("Finished.")


if __name__=="__main__":
    main(args.inputfile)
