import model_bagel
import pathlib

# import argparse
# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument("|epochs", type=int, default=50)
# parser.add_argument("|train_rate", type=float, default=0.49)
# parser.add_argument("|valid_rate", type=float, default=0.21)
# parser.add_argument("|test_rate", type=float, default=0.3)
# args = parser.parse_args()
# print('epochs: ', args.epochs, ' / train_rate: ', args.train_rate, ' / valid_rate: ', args.valid_rate, ' / test_rate: ', args.test_rate)



def predict():
    model_bagel.utils.mkdirs(output_path)
    file_list = model_bagel.utils.file_list(input_path)

    for file in file_list:
        print('loading test data')
        kpi = model_bagel.utils.load_kpi(file)
        print('loaded test data')
        print(f'KPI: {kpi.name}')
        kpi.complete_timestamp()

        model = model_bagel.Bagel()
        anomaly_scores = model.predict(kpi)

        for scores in anomaly_scores:
            if scores > 0.5:
                print('result: ','1')
            else:
                print('result: ','0')


if __name__ == '__main__':
    # epochs = args.epochs
    input_path = pathlib.Path('test')
    output_path = pathlib.Path('out').joinpath('model_bagel')
    predict()