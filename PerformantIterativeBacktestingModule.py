import os.path
from datetime import datetime, timezone
from itertools import product
import polars as pl
from polars import col
import numpy as np
import matplotlib.pyplot as plt
import BinanceClient as client
from IPython.display import display, Markdown
plt.style.use("seaborn")


class IterativeBacktesting ():
    units = 0
    trades = 0
    position = 0
    performance = 0.0
    trading_costs = []
    profits = []
    losses = []
    success_rate = 0
    data: pl.DataFrame = None
    trading_fee_percentage = 0.0002
    result_cols_schema = [
        ("date", pl.Utf8),
        ("windows", pl.Utf8),
        ("upper_limit", pl.Int32),
        ("reversal_window_index", pl.Int32),
        ("performance", pl.Float64),
        ("buy_and_hold_perf_diff", pl.Float64),
        ("success_rate", pl.Float64),
        ("trades", pl.Float64),
        ("trading_costs_sum", pl.Float64),
        ("profits_sum", pl.Float64),
        ("losses_sum", pl.Float64),
        ("cumulative_returns", pl.Float64),
        ("mean_returns", pl.Float64),
        ("risk", pl.Float64),
    ]

    def __init__(
        self,
        symbols: str,
        date: str,
        limit=720,
        amount=1000,
        use_spread=True,
        margin=0.5,
        leverage=1,
        stop_loss=None,
        take_gain=None,
        log_level=0
    ):
        self.symbols = symbols
        self.date = date
        self.initial_balance = amount
        self.current_balance = amount
        self.current_nav = self.initial_balance
        self.use_spread = use_spread
        self.client = client.Manager()
        self.log_level = log_level
        self.base_data_files_directory = "./data"
        self.parquet_data_directory = "./tests/parquets"
        self.loaded_symbols = []
        self.buy_and_hold_performance = 0
        self.leverage = leverage
        self.stop_loss = stop_loss
        self.take_gain = take_gain
        for symbol in symbols:
            if symbol in self.loaded_symbols:
                continue
            self.load_data(symbol, date, limit)
            self.loaded_symbols.append(symbol)
        self.data.sort("start_time")
        self.set_buy_and_hold_performance(symbols[1])

    def load_data(self,
                  symbol: str,
                  date: str,  # dd-mm-yyyy
                  limit: int = 720  # binance
                  ):

        symbol_data_directory = "{}/{}".format(
            self.base_data_files_directory, symbol)
        data_directory_exists = os.path.isdir(symbol_data_directory)
        if data_directory_exists is False:
            if self.log_level > 0:
                print("directory {} for storing {} data is missing, creating it...".format(
                    symbol_data_directory, symbol))
            os.mkdir(symbol_data_directory)

        filename = "{}.csv".format(date)
        file_path = "{}/{}".format(symbol_data_directory, filename)
        file_exists = os.path.isfile(file_path)
        if file_exists:
            if self.log_level > 1:
                print("file {} was found, importing data to DataFrame".format(filename))
            self.read_csv_file(symbol, file_path)
        else:
            if self.log_level > 1:
                print("{} csv file not found at {}, downloading data from binance".format(
                    filename, file_path))
            self.download_and_save_data(symbol, limit, date, filename)
        # self.set_buy_and_hold_performance()
        if symbol == self.symbols[0]:
            self.set_market_trend(symbol)

    def get_symbol_ohlc_cols(self, symbol: str):
        open_col = "{}_open".format(symbol)
        high_col = "{}_high".format(symbol)
        low_col = "{}_low".format(symbol)
        close_col = "{}_close".format(symbol)
        return [open_col, high_col, low_col, close_col]

    def read_csv_file(self, symbol: str, file_path: str):
        if self.data is None:
            self.set_empty_data()
        open_col, high_col, low_col, close_col = self.get_symbol_ohlc_cols(
            symbol)
        df = pl.read_csv(
            file_path,
            try_parse_dates=True,
            # schema=[("date", pl.Date), (open_col, pl.Float32), (high_col, pl.Float32), (low_col, pl.Float32), (close_col, pl.Float32),],
            columns=[
                "start_time", open_col, high_col, low_col, close_col
            ],
        )

        self.join_df_to_data(df)

    def set_empty_data(self):
        self.data = pl.DataFrame(
            schema=[("start_time", pl.Datetime)]
        )

    def join_df_to_data(self, df: pl.DataFrame):
        self.data = self.data.join(df, on="start_time", how="outer")

    def concat_df_to_data(self, df: pl.DataFrame):
        self.data = pl.concat([self.data, df], how="diagonal")

    def download_and_save_data(self, symbol: str, limit: int, date: str, filename: str):
        timestamp_intervals = self.date_to_timestamp_intervals(date, limit)
        # print("timestamp_intervals", timestamp_intervals)
        if self.data is None:
            self.set_empty_data()

        open_col, high_col, low_col, close_col = self.get_symbol_ohlc_cols(
            symbol)

        schema = {
            "start_time": pl.Utf8,
            open_col: pl.Float64,
            high_col: pl.Float64,
            low_col: pl.Float64,
            close_col: pl.Float64
        }
        download_schema = {
            "start_time": pl.Utf8,
            open_col: pl.Float64,
            high_col: pl.Float64,
            low_col: pl.Float64,
            close_col: pl.Float64,
            "volume": pl.Float64, "close_time": pl.Utf8, "quote_asset_volume": pl.Float64,
            "number_of_trades": pl.Float64, "taker_buy_base_asset_volume": pl.Float64,
            "taker_buy_quote_asset_volume": pl.Float64, "unused_field_ignore": pl.Utf8
        }

        df = pl.DataFrame(
            schema=schema
        )
        for (i, _) in enumerate(timestamp_intervals):
            if i == 0:
                continue

            start = timestamp_intervals[i - 1] * 1000
            end = timestamp_intervals[i] * 1000
            # print("start/end", start, end)
            query = self.client.get_kline(
                symbol=symbol,
                start=start,
                end=end,
                limit=limit
            )
            array = np.array(query)
            iter_df = pl.DataFrame(array, schema=download_schema)

            iter_df = iter_df.drop(columns=["volume", "close_time", "quote_asset_volume", "number_of_trades",
                                            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "unused_field_ignore"])
            df = pl.concat([df, iter_df], how="diagonal")

        df = self.parse_datetime(df)
        save_directory = "{}/{}/{}".format(
            self.base_data_files_directory, symbol, filename)
        df.write_csv(save_directory)
        self.join_df_to_data(df)

    def parse_datetime(self, df: pl.DataFrame):
        # return df.with_columns(
        #     col('start_time').str.strptime(
        #         datatype = pl.Date,
        #         fmt = '%d-%m-%Y %H:%M',
        #     ).keep_name()
        # )
        return df.with_columns(
            ((col('start_time') + '000').cast(pl.Datetime) -
             pl.duration(hours=3)).keep_name()
            # .str.strptime(pl.Datetime, fmt='%Y-%m-%d %H:%M').cast(pl.Datetime).keep_name()
        )

    def set_market_trend(self, symbol: str, short_span=21, long_span=50):
        _, _, _, close_col = self.get_symbol_ohlc_cols(symbol)
        ema_short_col = "{}_ema_s".format(symbol)
        ema_long_col = "{}_ema_l".format(symbol)
        trend_col = "{}_trend".format(symbol)
        self.data = self.data.with_columns(
            (col(close_col).ewm_mean(span=short_span,
             adjust=False)).alias(ema_short_col)
        )
        self.data = self.data.with_columns(
            (col(close_col).ewm_mean(span=long_span,
             adjust=False)).alias(ema_long_col)
        )
        self.data = self.data.with_columns(
            pl.when(col(ema_long_col) < col(ema_short_col)).then(
                "bullish").otherwise("bearish").alias(trend_col)
        )

    def get_values(self, symbol: str, bar: int):
        date = self.data[bar, "start_time"]
        _, _, _, close_col = self.get_symbol_ohlc_cols(symbol)
        price = self.data[bar, close_col]
        if date is None or price is None:
            return None, None
        price = round(self.data[bar, close_col], 6)
        return date, price

    def print_current_balance(self, symbol: str, bar: int):
        date, _ = self.get_values(symbol, bar)
        if self.log_level > 0:
            print("{} | Current Balance: {}".format(
                date, round(self.current_balance, 2)))

    def check_position(self, anchor_symbol: str, traded_symbol: str, bar: int):
        _, price = self.get_values(traded_symbol, bar)
        updated_nav = self.calculate_nav(price)
        if updated_nav > self.current_nav and self.take_gain is not None:
            is_gain_to_be_taken = (updated_nav / self.current_nav) > (1+ (self.take_gain / 100))
            if is_gain_to_be_taken:
                self.close_current_position(anchor_symbol, traded_symbol, bar)
        elif self.current_nav > updated_nav and self.stop_loss is not None:
            is_loss_to_be_stopped = (updated_nav / self.current_nav) > (1+ (self.stop_loss / 100))
            if is_loss_to_be_stopped:
                self.close_current_position(anchor_symbol, traded_symbol, bar)

    def close_long(self, anchor_symbol: str, traded_symbol: str, bar: int):
        if self.position != 1:
            return False
        return self.sell_instrument(anchor_symbol, traded_symbol, bar, units=self.units,
                                    label="Closing Long Position")

    def close_short(self, anchor_symbol: str, traded_symbol: str, bar: int):
        if self.position != -1:
            return False
        return self.buy_instrument(anchor_symbol, traded_symbol, bar, units=-self.units,  # -self.units
                                   label="Closing Short Position")

    def go_long(self, anchor_symbol: str, traded_symbol: str, bar: int, units=None, amount=None):
        if self.position == 1:  # if current position is long, do nothing
            return True

        if self.position == -1:  # if current position is short, close short, then
            if self.close_short(anchor_symbol, traded_symbol, bar) is False:
                return False

        # take long position
        if units:
            return self.buy_instrument(anchor_symbol, traded_symbol, bar, units=units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            return self.buy_instrument(
                anchor_symbol, traded_symbol, bar, amount=amount)
        return True

    def go_short(self, anchor_symbol: str, traded_symbol: str, bar: int, units=None, amount=None):
        if self.position == -1:  # if current position is short, do nothing
            return True
        if self.position == 1:  # if current position is long, close long, then
            if self.close_long(anchor_symbol, traded_symbol, bar) is False:
                return False

        # take short position
        if units:
            return self.sell_instrument(
                anchor_symbol, traded_symbol, bar, units=units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            return self.sell_instrument(
                anchor_symbol, traded_symbol, bar, amount=amount)
        return True

    def buy_instrument(self, anchor_symbol: str, traded_symbol: str, bar: int, units=None, amount=None, label="Taking Long Position"):
        date, feeless_price = self.get_values(traded_symbol, bar)
        if date is None or feeless_price is None:
            return False
        price = feeless_price
        trading_fee = 0
        if self.use_spread:
            trading_fee = price * self.trading_fee_percentage
            price += trading_fee  # ask price
        if amount is not None:
            units = (amount * self.leverage / price)
        result = self.calculate_result(price)

        transaction_cost = abs(self.current_balance *
                               self.trading_fee_percentage) * self.use_spread
        self.current_balance -= units * price
        self.units += units
        self.trades += 1

        self.trading_costs.append(transaction_cost)

        self.append_results(result)
        # if ~np.isclose(abs(result), transaction_cost, equal_nan=False):
        if self.log_level > 1:
            self.display_transaction_log(
                date, "Buying", units, anchor_symbol, traded_symbol, label, price, result, transaction_cost, bar
            )
        return True

    def sell_instrument(self, anchor_symbol: str, traded_symbol: str, bar: int, units=None, amount=None, label="Taking Short Position"):
        date, feeless_price = self.get_values(traded_symbol, bar)
        if date is None or feeless_price is None:
            return False
        trading_fee = 0
        price = feeless_price
        if self.use_spread:
            trading_fee = feeless_price * self.trading_fee_percentage
            price -= trading_fee  # bid price
        if amount is not None:
            units = (amount * self.leverage / price)
        result = self.calculate_result(price)

        self.current_balance += units * price
        self.units -= units
        self.trades += 1
        transaction_cost = abs(self.current_balance *
                               self.trading_fee_percentage) * self.use_spread

        self.trading_costs.append(transaction_cost)

        self.append_results(result)
        # if ~np.isclose(abs(result), transaction_cost, equal_nan=False):
        if self.log_level > 1:
            self.display_transaction_log(
                date, "Selling", units, anchor_symbol, traded_symbol, label, price, result, transaction_cost, bar
            )

        return True

    def append_results(self, result: float):
        if result > 0:
            self.profits.append(result)
            return
        self.losses.append(result)

    def close_current_position(self, anchor_symbol: str,  traded_symbol: str, bar: int):
        if self.position == 0:
            return
        date, feeless_price = self.get_values(traded_symbol, bar)
        if date is None or feeless_price is None:
            return
        if self.log_level > 0:
            print(100 * "-")
            print("{} | +++ CLOSING POSITION ON {} +++".format(date, traded_symbol))
        if self.position == 1:
            self.close_long(anchor_symbol, traded_symbol, bar)
        elif self.position == -1:
            self.close_short(anchor_symbol, traded_symbol, bar)

    def calculate_nav(self, price):
        return self.current_balance + self.units * price * self.leverage

    def calculate_result(self, price: float):
        updated_nav = self.calculate_nav(price)
        diff = updated_nav - self.current_nav
        self.current_nav = updated_nav
        return diff

    def clear_transactions(self):
        self.trades = 0
        self.position = 0
        self.trading_costs = []
        self.profits = []
        self.losses = []
        self.current_balance = self.initial_balance

    def set_buy_and_hold_performance(self, traded_symbol: str):
        df_count = self.data.select(pl.count())[0, 'count']
        _, bnh_initial_price = self.get_values(traded_symbol, 0)
        _, bnh_final_price = self.get_values(traded_symbol, df_count - 1)
        if self.use_spread:
            bnh_initial_price = bnh_initial_price * \
                (1 + self.trading_fee_percentage)
            bnh_final_price = bnh_final_price * \
                (1 - self.trading_fee_percentage)
        bnh_units = self.initial_balance / bnh_initial_price
        bnh_final_balance = bnh_units * bnh_final_price

        self.buy_and_hold_performance = round(
            ((bnh_final_balance - self.initial_balance) / self.initial_balance * 100), 2)

    def set_performance(self):
        traded_symbol = self.symbols[1]
        _, _, _, price_col = self.get_symbol_ohlc_cols(traded_symbol)

        self.data = self.data.with_columns([
            ((col(price_col) / (col(price_col).shift(1))).log()
             ).alias("buy_and_hold_performance"),
            (pl.when(
                (col("position") != col("position").shift(1)) &
                col("position").shift(1).is_not_null()
            ).then(1).otherwise(0).alias("trade"))
        ]
        )

        self.data = self.data.with_columns(
            (col("position").shift(1) *
             col("buy_and_hold_performance")).alias("returns"),
        )

        self.data = self.data.with_columns(
            (col("returns").cumsum().exp()).alias("c_returns")
        )

        cumulative_returns = self.data[-1, "c_returns"]
        performance = (cumulative_returns - 1) * 100

        buy_and_hold_perf = self.data[-1, "buy_and_hold_performance"]
        buy_and_hold_perf_diff = performance - buy_and_hold_perf

        # performance_df = self.get_positioned_data("returns", "position")
        trades = self.data.filter((col("trade") == 1)).select(
            pl.count())[0, "count"]

        self.data = self.data.with_columns((
            col("trade").sign().cumsum().alias('sesh')
        ))

        sesh_df = self.data.groupby(
            "sesh"
        ).agg(
            col(price_col).mean(),
            col("start_time").first().alias("start"),
            col("start_time").last().alias("end"),
            col("returns").cumsum().sum(),
        )
        profits_sum = 0

        success_trades = sesh_df.filter(
            col("returns").sum() > 0).select(pl.count())[0, "count"]
        trading_costs_sum = sesh_df.select(
            (((self.initial_balance * col("returns")) / col(price_col))
             * self.trading_fee_percentage).sum()
        )[0, 0]

        losses_sum = 0

        success_rate = 100 * abs(success_trades / trades) if trades > 0 else 0
        cumulative_returns = performance
        data = self.get_positioned_data("returns", "position")[:, "returns"]
        mean_returns = self.mean_returns(data, "returns")
        risk = self.risk_returns(data, "returns")

        return round(performance, 2), round(buy_and_hold_perf_diff, 2), round(success_rate, 2), trades, round(trading_costs_sum, 2), round(profits_sum, 2), round(losses_sum, 2), round(cumulative_returns, 4), round(mean_returns, 4), round(risk, 4)

        # success_rate =

    def set_iteractive_performance(self, date: str, traded_symbol: str, label: str):
        _, _, _, price_col = self.get_symbol_ohlc_cols(traded_symbol)
        self.data = self.data.with_columns(
            (
                col("position").shift(1) *
                ((col(price_col) / (col(price_col).shift(1))).log())
            )
            .alias("returns")
        )

        performance = round(((self.current_balance - self.initial_balance) /
                             self.initial_balance * 100), 2
                            )

        trades = self.trades
        trading_costs_sum = round(abs(sum(self.trading_costs)), 2)
        profits_sum = round(abs(sum(self.profits)), 2)
        losses_sum = round(abs(sum(self.losses)), 2)
        profits_count = len(self.profits)
        losses_count = len(self.losses)
        transactions_count = profits_count + losses_count
        success_rate = round((100 *
                              profits_count / transactions_count), 2) if transactions_count > 0 else 0

        data = self.get_positioned_data("returns", "position")[:, "returns"]
        mean_returns = self.mean_returns(data, "returns")
        risk = self.risk_returns(data, "returns")
        cumulative_returns = self.cumulative_results(data, "returns")

        buy_and_hold_perf_diff = round(
            performance - self.buy_and_hold_performance, 2)

        if self.log_level > 0:
            print(
                "{} | {} Perf. (%) = {} | Diff. to Buy & Hold (%) = {} | success rate (%) = {}".format(
                    date,
                    label,
                    performance,
                    buy_and_hold_perf_diff,
                    success_rate,
                )
            )
            print(
                "{} | trades count = {} | trading costs = {} | profits sum = {} | losses sum = {}".format(
                    date,
                    trades,
                    "{0:.2f}".format(trading_costs_sum),
                    "{0:.2f}".format(profits_sum),
                    "{0:.2f}".format(losses_sum)
                )
            )
            print(
                "{} | cumulative returns = {} | avg. returns (Σ mean) = {} | risk (Σ std) = {}".format(
                    date,
                    "{0:.4f}".format(cumulative_returns),
                    "{0:.4f}".format(mean_returns),
                    "{0:.4f}".format(risk)
                )
            )
        return performance, buy_and_hold_perf_diff, success_rate, trades, trading_costs_sum, profits_sum, losses_sum, cumulative_returns, mean_returns, risk

    def get_positioned_data(self, returns_col: str, position_col: str):
        return self.data.filter(col(position_col) != 0)

    # .apply(abs)
    def mean_returns(self, data: pl.Series, returns_col: str):
        result = data.select(col(returns_col).mean() * pl.count())
        result = result.drop_nulls()
        if result is None or result.is_empty():
            return 0
        return round(result[0, 0], 6)

    def risk_returns(self, data: pl.Series, returns_col: str):
        result = data.select(col(returns_col).std() * pl.count())
        result = result.drop_nulls()
        if result is None or result.is_empty():
            return 0
        return round(result[0, 0], 6)

    def cumulative_results(self, data: pl.Series, returns_col: str):
        result = data.select(col(returns_col).cumsum())
        result = result.drop_nulls()
        if result is None or result.is_empty():
            return 0
        return round(result[-1, 0], 6)

    def print_current_position_value(self, symbol: str, bar: int):
        date, price = self.get_values(symbol, bar)
        cpv = self.units * price
        print("{} | Current Position Value = {}".format(date, round(cpv, 2)))

    def date_to_timestamp_intervals(self, date_str: str, limit: int, granularity_in_mins=1):
        day, month, year = date_str.split("-")
        date = datetime(int(year), int(month), int(day))  # tzinfo=None = UTC
        timestamp_start = int(date.timestamp())
        day_in_mins = 86_400
        timestamp_end = timestamp_start + day_in_mins
        seconds_in_mins = 60
        step = int(limit * seconds_in_mins / granularity_in_mins)
        daily_range = range(timestamp_start, timestamp_end, step)
        return [*daily_range, timestamp_end]

    def display_transaction_log(
            self,
            date: str,
            transaction: str,
            units: int,
            anchor_symbol: str,
            traded_symbol: str,
            label: str,
            price: int,
            result: int,
            transaction_cost: int,
            bar: int
    ):
        display(
            Markdown(
                "{} | {} {} {} | price = {} | {} market is {} | transaction cost = {} | <i>{}</i> {}".format(
                    date,
                    transaction,
                    "{0:.4f}".format(abs(round(units, 4))),
                    traded_symbol,
                    "{0:.6f}".format(round(price, 6)),
                    anchor_symbol,
                    self.cstr("🐮&nbsp;&nbsp; Bullish ", "lightgreen") if self.data[bar, anchor_symbol+"_trend"] == "bullish" else self.cstr(
                        "🐻 Bearish", "lightcoral"),
                    self.cstr("{0:.4f}".format(transaction_cost), "coral"),
                    label,
                    ""
                    if "Taking" in label
                    else
                    "| <b>{} = {} </b>".format(
                        self.cstr("PROFIT", "#bada55")
                        if result > 0
                        else self.cstr("LOSS", "orangered"),
                        "{0:.4f}".format(round(abs(result), 4))
                    ),

                )
            )
        )

    def cstr(self, s, color="black"):
        return '<span style="color:{}">{}</span>'.format(color, s)
