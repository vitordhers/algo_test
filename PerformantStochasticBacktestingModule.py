from PerformantIterativeBacktestingModule import IterativeBacktesting
import polars as pl
from polars import col


class StochasticBacktesting(IterativeBacktesting):
    has_granularity_window = []

    def __init__(self, symbols: list, date: str, limit=720, amount=1000, use_spread=True, log_level=0, avoid_loss_lock=False):
        super().__init__(symbols, date, limit, amount,
                         use_spread, log_level, avoid_loss_lock)
        self.has_granularity_window = []

    def set_stochastic_data(self, symbol: str, data: pl.DataFrame, k_window=14, d_window=3, granularity=1):
        k_column_label = "{}_K%_{}".format(symbol, granularity)
        d_column_label = "{}_D%_{}".format(symbol, granularity)

        _, high_col, low_col, close_col = self.get_symbol_ohlc_cols(symbol)

        data = data.with_columns(
            (
                (100 *
                 (col(close_col) - col(low_col).rolling_min(k_window))
                 /
                 (col(high_col).rolling_max(k_window) -
                  col(low_col).rolling_min(k_window))).round(2)
            ).alias(k_column_label)
        )

        data = data.with_columns(
            (col(k_column_label).rolling_mean(d_window).round(2)
             ).alias(d_column_label)
        )

        data = data.upsample(time_column='start_time',
                             every="1m").fill_null(strategy="forward")

        stochastic_columns_df = data.select([k_column_label,
                                             d_column_label])

        self.data = pl.concat(
            [self.data, stochastic_columns_df], how="horizontal")
        self.has_granularity_window.append(granularity)
        self.data.write_csv("./tests/stochastic_data.csv")

    def set_multi_stochastic_windows(self, symbol: str, windows: list):
        open_col, high_col, low_col, close_col = self.get_symbol_ohlc_cols(
            symbol)
        for window in windows:
            if window in self.has_granularity_window:
                continue
            resample_period = '{}m'.format(window)
            # check resample params

            df_count = self.data.select(pl.count())[0, 'count']
            if self.data.is_empty() or df_count == 1:
                print('FAULTY STOCHASTIC, windows={}, window={}'.format(
                    windows, window))
                print(self.data)
                return

            resampled_data = self.data.fill_null(strategy="forward").upsample(
                time_column='start_time', every=resample_period
            ).groupby("start_time").agg([
                (col(open_col).first()).alias(open_col),
                (col(high_col).max()).alias(high_col),
                (col(low_col).min()).alias(low_col),
                (col(close_col).last()).alias(close_col),
            ])
            self.set_stochastic_data(symbol, resampled_data, 14, 3, window)
