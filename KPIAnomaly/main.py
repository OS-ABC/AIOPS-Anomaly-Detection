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



def main():
    model_bagel.utils.mkdirs(output_path)
    file_list = model_bagel.utils.file_list(input_path)

    for file in file_list:
        kpi = model_bagel.utils.load_kpi(file)
        print(f'KPI: {kpi.name}')
        kpi.complete_timestamp()
        train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
        train_kpi, mean, std = train_kpi.standardize()
        valid_kpi, _, _ = valid_kpi.standardize(mean=mean, std=std)
        test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)

        model = model_bagel.Bagel()
        print(model)
        model.fit(kpi=train_kpi.use_labels(0.), validation_kpi=valid_kpi, epochs=5, verbose=1)
        anomaly_scores = model.predict(test_kpi)

        results = model_bagel.testing.get_test_results(labels=test_kpi.labels,
                                                 scores=anomaly_scores,
                                                 missing=test_kpi.missing,
                                                 window_size=120)
        stats = model_bagel.testing.get_kpi_stats(kpi, test_kpi)
        print('Metrics')
        print(f'precision: {results.get("precision"):.3f} - '
              f'recall: {results.get("recall"):.3f} - '
              f'f1score: {results.get("f1score"):.3f}\n')

        with open(output_path.joinpath(f'{kpi.name}.txt'), 'w') as output:
            output.write(f'kpi_name={kpi.name}\n\n'

                         '[result]\n'
                         f'threshold={results.get("threshold")}\n'
                         f'precision={results.get("precision"):.3f}\n'
                         f'recall={results.get("recall"):.3f}\n'
                         f'f1_score={results.get("f1score"):.3f}\n\n'

                         '[overall]\n'
                         f'num_points={stats[0].num_points}\n'
                         f'num_missing_points={stats[0].num_missing}\n'
                         f'missing_rate={stats[0].missing_rate:.6f}\n'
                         f'num_anomaly_points={stats[0].num_anomaly}\n'
                         f'anomaly_rate={stats[0].anomaly_rate:.6f}\n\n'

                         '[test]\n'
                         f'num_points={stats[1].num_points}\n'
                         f'num_missing_points={stats[1].num_missing}\n'
                         f'missing_rate={stats[1].missing_rate:.6f}\n'
                         f'num_anomaly_points={stats[1].num_anomaly}\n'
                         f'anomaly_rate={stats[1].anomaly_rate:.6f}\n')


if __name__ == '__main__':
    # epochs = args.epochs
    input_path = pathlib.Path('data')
    output_path = pathlib.Path('out').joinpath('model_bagel')
    main()