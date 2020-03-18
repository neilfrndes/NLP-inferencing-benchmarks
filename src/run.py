import argparse
import logging
import multiprocessing
import torch

from pathlib import Path
from timeit import default_timer as timer
from transformers import pipeline

logger = logging.getLogger('__name__')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Generates a model pickle.')
parser.add_argument(
    '-m', '--model', type=str, default="sentiment-analysis",
    help="Name of the model file")
parser.add_argument(
    '-l', '--loops', type=int, default=5,
    help="Number of loops to run inferencing for")
parser.add_argument(
    '-p', '--processes', type=int, default=multiprocessing.cpu_count(),
    help="Number of cores to use for inferencing")

def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    mean = sum(time_list) / len(time_list) 
    variance = sum([((x - mean) ** 2) for x in time_list]) / len(time_list) 
    std_dev = variance ** 0.5
    return (mean, std_dev)

args = parser.parse_args()
if __name__=='__main__':

    # Load data
    file_path = Path.joinpath(
        Path(__file__).resolve().parent.parent, 'data', args.model+".txt")
    logging.info(f"Reading {file_path}")
    with open(file_path, 'r') as f:
        data = f.readlines()
    num_lines = len(data)
    logging.info(f"Read {num_lines} lines")

    # Initialize Model
    logging.info(f"Initializing pre trained model for {args.model}")
    model = pipeline(args.model)

    # Run Infernencing
    run_times = []
    inference_times = []
    logging.info("Starting Inferencing..")
    for loop in range(1, args.loops+1):
        
        logging.info(f"Running loop #{loop}..")
        start_time = timer()

        # Synchronous
        results = [model(sentence) for sentence in data]

        # Asynchronous
        # pool = multiprocessing.Pool(processes=1)#args.processes)
        # results = [pool.apply_async(model, args=(sentence,)) for sentence in data]
        # output = [p.get() for p in results]

        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time)
        inference_times.append(total_time*1000/num_lines)

        logging.debug(results)
        logging.info(f"Loop #{loop} took {total_time} seconds.")
        
    logging.info("Finished Inferencing.")
    mean, std_dev = calculate_stats(run_times)
    logging.info(f"Loops: {len(run_times)} | Time per loop {mean:.2f}s | Standard Deviation {std_dev:.2f}s")

    mean, std_dev = calculate_stats(inference_times)
    logging.info(f"Time per inference {mean:.2f}ms | Standard Deviation {std_dev:.2f}ms")



    

