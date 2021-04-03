import os
import bagel


def main():
    bagel.utils.mkdirs(OUTPUT)
    file_list = bagel.utils.file_list(INPUT)

    for file in file_list:
        kpi = bagel.utils.load_kpi(file)
        print(f'KPI: {kpi.name}')
        kpi.complete_timestamp()
        train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
        train_kpi, mean, std = train_kpi.standardize()
        valid_kpi, _, _ = valid_kpi.standardize(mean=mean, std=std)
        test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)

        model = bagel.Bagel()
        model.fit(kpi=train_kpi.use_labels(0.), validation_kpi=valid_kpi, epochs=EPOCHS, verbose=1)
        anomaly_scores = model.predict(test_kpi)

        results = bagel.testing.get_test_results(labels=test_kpi.labels,
                                                 scores=anomaly_scores,
                                                 missing=test_kpi.missing)
        stats = bagel.testing.get_kpi_stats(kpi, test_kpi)
        print('Metrics')
        print(f'precision: {results.get("precision"):.3f} - '
              f'recall: {results.get("recall"):.3f} - '
              f'f1score: {results.get("f1score"):.3f}\n')

        with open(f'{os.path.join(OUTPUT, kpi.name)}.txt', 'w') as output:
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
    EPOCHS = 50
    INPUT = 'data'
    OUTPUT = os.path.join('out', 'bagel')
    main()
