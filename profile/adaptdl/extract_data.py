from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm


def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-path', type=str, default='./result/VGG/True/200', help='Tensorboard event files or a single tensorboard file location')
    parser.add_argument('--ex-path', type=str, default='./result/VGG/true.csv', help='location to save the exported data')

    args = parser.parse_args()
    event_data = event_accumulator.EventAccumulator(args.in_path)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    # print(keys)
    df = pd.DataFrame(columns=keys[7:])  # my first column is training loss per iteration, so I abandon it
    for key in tqdm(keys):
        # print(key)
        if key == 'Loss/Train' or key == 'Accuracy/Train' or key == 'Loss/Valid' or key == 'Accuracy/Valid':
            df[key] = pd.DataFrame(event_data.Scalars(key)).value

    df.to_csv(args.ex_path)

    print("Tensorboard data exported successfully")


if __name__ == '__main__':
    main()