import datetime
import json
import logging
import math
from zoneinfo import ZoneInfo

import pandas as pd
from entsoe import EntsoePandasClient
from nordpool.elspot import Prices as NordpoolPrices
from requests import get, post

from dao.lib.db_manager import DBmanagerObj


def parse_entsoe_prices(data: pd.Series) -> list[tuple[int, float]]:
    """Convert ENTSO-E's price series (€/MWh) to DAO's €/kWh pairs."""
    if data.empty:
        return []
    indexed = data.reset_index()
    prices = []
    for row in indexed.itertuples():
        prices.append((int(datetime.datetime.timestamp(row[1])), row[2] / 1000))
    return prices


def parse_easyenergy_prices(
    data: list[dict], time_zone: ZoneInfo
) -> list[tuple[int, float]]:
    """Convert the EasyEnergy tariff response to timestamp/value pairs."""
    prices = []
    for row in data:
        timestamp = datetime.datetime.fromisoformat(row["Timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=time_zone)
        else:
            timestamp = timestamp.astimezone(time_zone)
        prices.append((int(timestamp.timestamp()), row["TariffReturn"]))
    return prices


def parse_tibber_prices(data: dict) -> list[tuple[int, float]]:
    """Convert the Tibber GraphQL price response to timestamp/value pairs."""
    subscription = data["data"]["viewer"]["homes"][0]["currentSubscription"]
    price_info = subscription["priceInfo"]
    nodes = (
        price_info["today"]
        + price_info["tomorrow"]
        + subscription["priceInfoRange"]["nodes"]
    )
    prices = []
    for node in nodes:
        dt = datetime.datetime.strptime(
            node["startsAt"], "%Y-%m-%dT%H:%M:%S.%f%z"
        )
        prices.append((int(dt.timestamp()), float(node["energy"])))
    return prices


def parse_nordpool_prices(
    data: dict, country: str, time_zone: ZoneInfo
) -> list[tuple[int, float]]:
    """Convert a Nordpool daily response from €/MWh to DAO price pairs."""
    prices = []
    for value_row in data["areas"][country]["values"]:
        start = value_row["start"]
        if start.tzinfo is None:
            start = start.replace(tzinfo=time_zone)
        else:
            start = start.astimezone(time_zone)
        value = value_row["value"]
        if value == float("inf"):
            continue
        prices.append((int(start.timestamp()), value / 1000))
    return prices


class DaPrices:
    def __init__(
        self, config,
            db_da: DBmanagerObj,
            country: str = None,
            secrets: dict = None,
            time_zone: str = "Europe/Amsterdam"
    ):
        self.config = config
        self.db_da = db_da
        self._secrets = secrets or {}
        self.interval = str(config.interval or "1hour").lower()
        self.country = country if country is not None else "NL"
        self.time_zone = ZoneInfo(time_zone) if time_zone is not None else ZoneInfo("Europe/Amsterdam")

    def _timezoned_datetime(self, value: datetime.datetime) -> datetime.datetime:
        """Interpret naive datetimes as local time; convert tz aware ones."""
        if value.tzinfo is None:
            return value.replace(tzinfo=self.time_zone)
        return value.astimezone(self.time_zone)

    def get_prices(
        self, source, _start: datetime.datetime|None = None, _end: datetime.datetime|None = None
    ):
        if self.interval == "1hour":
            resolution = 60
        else:
            resolution = 15
        now = datetime.datetime.now(self.time_zone)

        if _start is None:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start = self._timezoned_datetime(_start)

        if _end is None:
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if now.hour < 12:
                end = end + datetime.timedelta(days=1)
            else:
                end = end + datetime.timedelta(days=2)
        else:
            end = self._timezoned_datetime(_end)

        present = self.db_da.get_time_border_record("da")
        if present is not None and _start is None:
            present = self._timezoned_datetime(present)
            present = present + datetime.timedelta(minutes=resolution)
            start = max(present, start)

            if present >= (end - datetime.timedelta(hours=1)):
                logging.info("Spot prices already present")
                return

        logging.debug(f"Day ahead tarieven ophalen voor {start} tot {end}")

        # Nordpool only returns one date per request, so fetch each date.
        prices = []

        if source.lower() == "nordpool":

            # Process each day in the range separately
            current_date = start
            while current_date < end:
                # Create a date range for one day (end of current day)
                current_end = current_date + datetime.timedelta(days=1)

                try:
                    daily_prices = self._get_prices_nordpool(resolution, current_date)
                    if daily_prices:
                        prices.extend(daily_prices)

                except Exception as ex:
                    logging.error(f"Error fetching data for {current_date.date()}: {ex}")

                current_date = current_end

        else:
            # For other sources that support range queries
            # day-ahead market prices (€/MWh)
            if source.lower() == "entsoe":
                prices = self._get_prices_entsoe(start, end)

            if source.lower() == "easyenergy":
                prices = self._get_prices_easyenergy(start, end) or []

            if source.lower() == "tibber":
                prices = self._get_prices_tibber(start) or []

        end_dt = end - datetime.timedelta(minutes=resolution)
        if len(prices) == 0:
            logging.error(
                f"Geen data van {source} voor het gevraagde bereik "
                f"{start} tot en met {end_dt}"
            )
        else:
            last_time = max(int(timestamp) for timestamp, _ in prices)
            last_time_dt = datetime.datetime.fromtimestamp(
                last_time, tz=end.tzinfo
            )
            if last_time_dt < end_dt:
                logging.warning(
                    f"Geen data van {source} tussen {last_time_dt} en {end_dt}"
                )

            df_db = pd.DataFrame(prices, columns=["time", "value"])
            df_db["code"] = "da"
            df_db = df_db[["time", "code", "value"]]
            timestamps = df_db["time"].astype(int)
            delete_start = self._timezoned_datetime(datetime.datetime.fromtimestamp(
                int(timestamps.min()), tz=datetime.timezone.utc
            ))
            delete_end = self._timezoned_datetime(datetime.datetime.fromtimestamp(
                int(timestamps.max()), tz=datetime.timezone.utc
            ))
            self.db_da.deleterange("da", delete_start, delete_end)
            self.db_da.savedata(df_db)

    def _get_prices_entsoe(self, start, end):
        _ak = self.config.prices.entsoe_api_key
        api_key = _ak.resolve(self._secrets) if _ak is not None else None
        client = EntsoePandasClient(api_key=api_key)
        da_prices = pd.Series()
        try:
            da_prices = client.query_day_ahead_prices(
                self.country, start=start, end=end
            )
        except Exception as ex:
            logging.error(ex)

        if len(da_prices.index) > 0:
            logging.info(
                f"Day ahead prijzen van Entsoe: \n{da_prices.to_string(index=False)}"
            )
        return parse_entsoe_prices(da_prices)

    def _get_prices_tibber(self, date):
        now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
        get_ts = date.timestamp()
        count = 1 + math.ceil((now_ts - get_ts) / 3600)
        if self.interval == "1hour":
            resolution = "HOURLY"
        else:
            resolution = "QUARTER_HOURLY"
            count = count * 4
            if count > 674:
                count = 674
                logging.warning(
                    "Je kunt met Tibber maximaal 7 dagen terug opvragen"
                )
        count = max(1, min(674, count))
        query = (
                "{ "
                '"query": '
                ' "{ '
                "  viewer { "
                "    homes { "
                "      currentSubscription { "
                "        priceInfo(resolution: " + resolution + "){ "
                                                                "          today { "
                                                                "            energy "
                                                                "            startsAt "
                                                                "          } "
                                                                "          tomorrow { "
                                                                "            energy "
                                                                "            startsAt "
                                                                "          } "
                                                                "        } "
                                                                "        priceInfoRange(resolution: "
                + resolution
                + ", last: "
                + str(count)
                + ") { "
                  "          nodes { "
                  "            energy "
                  "            startsAt "
                  "          } "
                  "        } "
                  "      } "
                  "    } "
                  "  } "
                  '}" '
                  "}"
        )

        logging.debug(query)
        _tibber = self.config.tibber
        _tok = _tibber.api_token
        api_token = _tok.resolve(self._secrets)
        url = _tibber.api_url or "https://api.tibber.com/v1-beta/gql"
        headers = {
            "Authorization": "Bearer " + api_token,
            "content-type": "application/json",
        }
        resp = post(url, headers=headers, data=query)
        tibber_dict = json.loads(resp.text)
        return parse_tibber_prices(tibber_dict)

    def _get_prices_easyenergy(self, start, end):
        # ophalen bij EasyEnergy
        # 2022-06-25T00:00:00
        startstr = self._timezoned_datetime(start).strftime("%Y-%m-%dT%H:%M:%S")
        endstr = self._timezoned_datetime(end).strftime("%Y-%m-%dT%H:%M:%S")
        url = (
                "https://mijn.easyenergy.com/nl/api/tariff/getapxtariffs?startTimestamp="
                + startstr
                + "&endTimestamp="
                + endstr
        )
        resp = get(url)
        logging.debug(resp.text)
        json_object = json.loads(resp.text)
        logging.info(f"Day ahead prijzen van Easyenergy: {json_object}")
        return parse_easyenergy_prices(json_object, self.time_zone)

    def _get_prices_nordpool(self, resolution, date):

        # ophalen bij Nordpool
        prices_spot = NordpoolPrices()

        try:
            act_spot_prices = prices_spot.fetch(
                areas=[self.country], end_date=date, resolution=resolution
            )
        except Exception as ex:
            logging.exception(ex)
            logging.error(f"Fout bij ophalen data van Nordpool voor {date}")
            return []
        if act_spot_prices is None:
            logging.error(f"Geen data van Nordpool voor {date}")
            return []

        logging.info(f"Day ahead prijzen van Nordpool:")
        return parse_nordpool_prices(act_spot_prices, self.country, self.time_zone)

    def extract_data_epexpredictor(self,
                                   data:dict,
                                   know_at:datetime.datetime,
                                   new_horizon:datetime.datetime
                                   )->pd.DataFrame:
        """
        :param data:
        {
        "knownUntil":"2026-09-21T23:45:00+02:00",
        "prices": [
            {
              "startsAt": "2026-09-16T23:45:00+02:00",
              "total": 16.851
            },
            {
              "startsAt": "2026-09-17T00:00:00+02:00",
              "total": 19.944
            },
            {
              "startsAt": "2026-09-17T00:15:00+02:00",
              "total": 18.678
            },
            ....
        ]
        :param know_at: laatste record met day ahead prijzen
        :param new_horizon: ophalen tot en met new horizon
        :return: dataframe with code, time, value
        """
        rows = data.get("prices")
        df_db = pd.DataFrame(columns=["time", "tijd", "code", "value"])
        for row in rows:
            dt = datetime.datetime.strptime(row["startsAt"], "%Y-%m-%dT%H:%M:%S%f%z")
            if know_at < dt <= new_horizon:
                time_stamp = int(dt.timestamp())
                value = float(row["total"])/100
                logging.info(f"{row} {dt} {time_stamp} {value}")
                df_db.loc[df_db.shape[0]] = [time_stamp, dt, "da", value]
        return df_db

    def extract_data_energypriceforecast_eu(self,
                                            data:dict,
                                            know_at:datetime.datetime,
                                            new_horizon:datetime.datetime
                                            )->pd.DataFrame:
        """
        {
          "api_version": "v1",
          "country": "NL",
          "currency": "EUR",
          "entries": [
            {
              "end": "2026-09-21T21:15:00Z",
              "source": "day_ahead",
              "start": "2026-09-21T21:00:00Z",
              "value": 0.19836
            },
            {
              "end": "2026-09-21T21:30:00Z",
              "source": "day_ahead",
              "start": "2026-09-21T21:15:00Z",
              "value": 0.18462
            },

        """
        rows = data.get('entries')
        df_db = pd.DataFrame(columns=["time", "tijd", "code", "value"])
        for row in rows:
            utc_dt_str = row["start"].replace("Z", "+00:00")
            utc_dt = datetime.datetime.fromisoformat(utc_dt_str)
            dt = utc_dt.astimezone(self.time_zone)
            if know_at < dt <= new_horizon:
                time_stamp = int(dt.timestamp())
                value = float(row["value"])
                logging.info(f"{row} {dt} {time_stamp} {value}")
                df_db.loc[df_db.shape[0]] = [time_stamp, dt, "da", value]
        return df_db

    def extract_data_dap(self,
                         data:dict,
                         know_at:datetime.datetime,
                         new_horizon:datetime.datetime
                         )->pd.DataFrame:
        """
        :param data:
        [
          {
            "time": "2026-09-22 00:00:00+02:00",
            "time_ts": 1790028000,
            "prediction": 0.1701457947
          },
          {
            "time": "2026-09-22 01:00:00+02:00",
            "time_ts": 1790031600,
            "prediction": 0.1664741486
          },
          {
            "time": "2026-09-22 02:00:00+02:00",
            "time_ts": 1790035200,
            "prediction": 0.1623867005
          },
        ....
        """
        df = pd.DataFrame.from_records(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.loc[df['time'] > know_at]
        df = df.loc[df['time'] <= new_horizon]
        df.rename(columns={"time":"time_dt", "time_ts":"time", "prediction":"value"}, inplace=True)
        df["code"] = "da"
        return df

    def extract_data_jeroen_nl(self,
                               data:dict,
                               know_at:datetime.datetime,
                               new_horizon:datetime.datetime
                               )->pd.DataFrame:

        """
        [
          {
            "datum_nl": "2026-10-15 00:00:00",
            "datum_utc": "2026-10-14 22:00:00",
            "prijs_excl_belastingen": "0,147734",
            "band_laag": "0,138766",
            "band_hoog": "0,206114",
            "voorspeld_op": "2026-10-13 05:00:00"
          },
          {
            "datum_nl": "2026-10-15 00:15:00",
            "datum_utc": "2026-10-14 22:15:00",
            "prijs_excl_belastingen": "0,140237",
            "band_laag": "0,131390",
            "band_hoog": "0,198617",
            "voorspeld_op": "2026-10-13 05:00:00"
          },
          {
            "datum_nl": "2026-10-15 00:30:00",
            "datum_utc": "2026-10-14 22:30:00",
            "prijs_excl_belastingen": "0,137321",
            "band_laag": "0,128478",
            "band_hoog": "0,195701",
            "voorspeld_op": "2026-10-13 05:00:00"
          }
          .....
        ]
        """
        df_db = pd.DataFrame(columns=["time", "tijd", "code", "value"])
        tz = pytz.timezone(self.time_zone)
        for row in data:
            utc_dt_str = row["datum_utc"]+"+00:00"
            utc_dt = datetime.datetime.fromisoformat(utc_dt_str)
            dt = utc_dt.astimezone(tz)
            if know_at < dt <= new_horizon:
                time_stamp = int(dt.timestamp())
                value_str=row["prijs_excl_belastingen"]
                value = float(value_str.replace(",", "."))
                logging.info(f"{row} {dt} {time_stamp} {value}")
                df_db.loc[df_db.shape[0]] = [time_stamp, dt, "da", value]
        return df_db

    def get_predicted_prices(self):
        source = self.config.prices.prediction.source
        api_url = self.config.prices.prediction.api
        _key = self.config.prices.prediction.api_key
        key = _key.resolve(self._secrets) if _key is not None else None
        extension = 96
        if extension == 0:
            logging.warning("predicted horizon extension is 0, no predicted prices are returned")
            return
        known_at = self._timezoned_datetime(self.db_da.get_time_border_record("da"))
        new_horizon = known_at + datetime.timedelta(hours=extension)
        now = datetime.datetime.now(self.time_zone)
        fetch_hours = math.ceil((new_horizon - now).total_seconds()/3600)
        api_url= api_url.replace("<hours>", str(fetch_hours))
        api_url= api_url.replace("<region>", self.country)
        if key is not None:
            api_url= api_url.replace("<key>", key)
        resp = get(api_url)
        if resp.status_code != 200:
            logging.error(f"No data from {source} off predict prices, "
                          f"statuscode: {resp.status_code},"
                          f"message: {resp.text}")
            return resp.status_code
        logging.debug(resp.text)
        json_object = json.loads(resp.text)
        extract_data_f = "extract_data_"+source
        df_db = getattr(self, extract_data_f)(json_object, known_at, new_horizon)
        self.db_da.savedata(df_db, tablename="prognoses")
        return 0
