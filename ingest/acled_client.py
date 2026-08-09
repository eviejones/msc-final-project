import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger("ACLED ingest")


# Ref: https://stackoverflow.com/questions/60435406/which-exception-should-be-raised-when-a-required-environment-variable-is-missing
class MissingEnvironmentVariable(Exception):
    pass


class AcledClient:
    PAGE_SIZE = 5000

    def __init__(self):
        load_dotenv()
        self.endpoint = "https://acleddata.com/api/acled/read?_format=json"

        try:
            self.username = os.environ["ACLED_USERNAME"]
            self.password = os.environ["ACLED_PASSWORD"]
            self.token_url = os.environ["ACLED_TOKEN_URL"]
        except KeyError as e:
            raise MissingEnvironmentVariable(
                f"Environment variable {e.args[0]} does not exist. Make sure it is saved in the .env file."
            )

        # Token is valid for 24 hours
        self.access_token = self._get_access_token()

    def _get_access_token(self):
        """Gets the access token required to call the API."""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
            "client_id": "acled",
        }
        response = requests.post(self.token_url, headers=headers, data=data, timeout=20)

        if response.status_code == 200:
            token_data = response.json()
            logger.info("Access token correctly retrieved.")
            return token_data["access_token"]
        raise Exception(
            f"Failed to get access token: {response.status_code} {response.text}"
        )

    def _validate_dates(self, start_date: str, end_date: str) -> None:
        """Validates that dates are in YYYY-MM-DD format and logically ordered."""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            if start > end:
                raise ValueError(
                    f"start_date ({start_date}) cannot be after end_date ({end_date})."
                )
        except ValueError as e:
            if "does not match format" in str(e) or "unconverted data" in str(e):
                raise ValueError(
                    "Dates must be provided in the format YYYY-MM-DD."
                ) from e
            raise

    def _validate_countries(self, countries: list[str] | str) -> None:
        """Validates that country names are non-empty strings with recognised characters."""
        if not countries:
            raise ValueError("The countries list cannot be empty.")

        if isinstance(countries, str):
            countries = [countries]

        for country in countries:
            if not isinstance(country, str) or not country.strip():
                raise ValueError(
                    f"Invalid country name provided: '{country}'. It must be a non-empty string."
                )

    def _build_params(self) -> dict:
        """Builds the parameter dictionary from passed attributes and formats it based on the API documentation.

        Documentation found at: https://acleddata.com/acled-api-documentation"""
        if isinstance(self.countries, str):
            self.countries = [self.countries]

        if self.event_types is not None and isinstance(self.event_types, str):
            self.event_types = [self.event_types]

        params = {
            "country": ":OR:country=".join(self.countries),
            "event_date": f"{self.start_date}|{self.end_date}",
            "event_date_where": "BETWEEN",
            "limit": self.PAGE_SIZE,
        }
        if self.event_types is not None:
            params["event_type"] = ":OR:event_type=".join(self.event_types)
        return params
    
    def _save_data(self, df: pd.DataFrame):
        """Saves the fetched data to a csv file."""
        countries_str = "_".join(country.lower().replace(" ", "_") for country in self.countries)
        filename = f"acled_data_{countries_str}.csv"
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists("data/acled"):
            os.makedirs("data/acled")
        df.to_csv(f"data/acled/{filename}", index=False)
        logger.info(f"Data saved to data/acled/{filename}.")
        
    def _read_data(self, countries: list[str]) -> pd.DataFrame:
        """Reads the saved data from a csv file if it exists."""
        countries_str = "_".join(country.lower().replace(" ", "_") for country in countries)
        filename = f"acled_data_{countries_str}.csv"
        filepath = f"data/acled/{filename}"
        if os.path.exists(filepath):
            logger.info(f"Reading data from {filepath}.")
            return pd.read_csv(filepath)
        else:
            logger.warning(f"No saved data found for {countries_str}.")
            return pd.DataFrame()

    def get_data(
        self,
        countries: list[str],
        start_date: str,
        end_date: str,
        *,
        event_types=None,
        force_download: bool = False,
    ):
        """Gets data for specified countries and dates from the ACLED API.

        Args:
            countries (list[str]): List of countries to get data from.
            start_date (str): Start date for pulling data. Should be in format YYYY-MM-DD.
            end_date (str): End date for pulling data. Should be in format YYYY-MM-DD
            event_types (list[str]): List of event types to pull data from. Event types must match ACLED event types.
            force_download (bool): If True, bypass any cached CSV and re-fetch from the API.

        Returns:
            pd.DataFrame: Dataframe containing all data from the ACLED API.
        """
        self._validate_dates(start_date, end_date)
        self._validate_countries(countries)

        self.countries = countries
        self.start_date = start_date
        self.end_date = end_date
        self.event_types = event_types

        if not force_download:
            cached_df = self._read_data(countries)
            if not cached_df.empty:
                return cached_df

        params = self._build_params()
        params["page"] = 1
        request_end = False
        r_dfs = []

        while not request_end:
            r = requests.get(
                url=self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                },
                params=params,
                timeout=20,
            )
            if r.ok:
                r_df = pd.DataFrame.from_dict(r.json()["data"])
                r_dfs.append(r_df)
                logger.info("Requesting data...")

                if len(r_df) < self.PAGE_SIZE:
                    request_end = True
                else:
                    params["page"] += 1
            else:
                raise requests.RequestException(
                    f"HTTP Code: {r.status_code}, Status: {r.reason}"
                )
        final_df = pd.concat(r_dfs)
        if final_df.empty:
            logger.warning("No data found for the given parameters.")
            return final_df 

        final_df["event_date"] = pd.to_datetime(final_df["event_date"])
        final_df["year_month"] = final_df["event_date"].dt.to_period("M")
        logger.info("All data successfully fetched.")
        self._save_data(final_df)
        return final_df
