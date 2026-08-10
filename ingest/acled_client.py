import os

import pandas as pd
import requests
from dotenv import load_dotenv

from utils.constants import FORCE_DOWNLOAD

from utils.logger import get_logger

logger = get_logger("AcledClient")


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

    def _validate_dates(self, start_date, end_date) -> None:
        """Validates that dates are parseable (str in YYYY-MM-DD or pd.Timestamp) and logically ordered."""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except (ValueError, TypeError) as e:
            raise ValueError(
                "Dates must be a string in YYYY-MM-DD format or a pd.Timestamp."
            ) from e

        if start > end:
            raise ValueError(
                f"start_date ({start_date}) cannot be after end_date ({end_date})."
            )

    def _validate_country(self, country: str) -> None:
        """Validates that the country name is a non-empty string."""
        if not isinstance(country, str) or not country.strip():
            raise ValueError(
                f"Invalid country name provided: '{country}'. It must be a non-empty string."
            )

    def _build_params(self) -> dict:
        """Builds the parameter dictionary from passed attributes and formats it based on the API documentation.

        Documentation found at: https://acleddata.com/acled-api-documentation"""
        if self.event_types is not None and isinstance(self.event_types, str):
            self.event_types = [self.event_types]

        params = {
            "country": self.country,
            "event_date": f"{self.start_date}|{self.end_date}",
            "event_date_where": "BETWEEN",
            "limit": self.PAGE_SIZE,
        }
        if self.event_types is not None:
            params["event_type"] = ":OR:event_type=".join(self.event_types)
        return params

    def _save_data(self, df: pd.DataFrame):
        """Saves the fetched data to a csv file."""
        country_str = self.country.lower().replace(" ", "_")
        filename = f"acled_{country_str}.csv"
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists("data/acled"):
            os.makedirs("data/acled")
        df.to_csv(f"data/acled/{filename}", index=False)
        logger.info(f"Data saved to data/acled/{filename}.")

    def _read_data(self, country: str) -> pd.DataFrame:
        """Reads the saved data from a csv file if it exists."""
        country_str = country.lower().replace(" ", "_")
        filename = f"acled_{country_str}.csv"
        filepath = f"data/acled/{filename}"
        if os.path.exists(filepath):
            logger.info(f"Reading data from {filepath}.")
            return pd.read_csv(filepath)
        else:
            logger.warning(f"No saved data found for {country_str}.")
            return pd.DataFrame()
        
        
    def _mark_conflict_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the ACLED dataframe and marks event as conflict (1) or not conflict (0).

        Conflict events are used to create the target Y.

        Args:
            df (pd.DataFrame): The full ACLED dataframe
        Returns:
            pd.DataFrame: The dataframe with an additional column 'conflict' with binary markers.
        Raises:
            ValueError: If 'sub_event_type' contains values not present in the mapping.
        """

        acled_subevent_mapping = {
            # BATTLES (Conflict)
            "Armed clash": 1,
            "Government regains territory": 1,
            "Non-state actor overtakes territory": 1,
            # EXPLOSIONS / REMOTE VIOLENCE (Conflict)
            "Air/drone strike": 1,
            "Chemical weapon": 1,
            "Remote explosive/landmine/IED": 1,
            "Shelling/artillery/missile attack": 1,
            "Suicide bomb": 1,
            "Grenade": 1,
            # VIOLENCE AGAINST CIVILIANS (Conflict)
            "Abduction/forced disappearance": 1,
            "Attack": 1,
            "Sexual violence": 1,
            # RIOTS (Conflict)
            "Mob violence": 1,
            "Violent demonstration": 1,
            # PROTESTS (Non-conflict)
            "Excessive force against protesters": 0,
            "Peaceful protest": 0,
            "Protest with intervention": 0,
            # STRATEGIC DEVELOPMENTS (Non-conflict)
            "Agreement": 0,
            "Arrests": 0,
            "Change to group/activity": 0,
            "Disrupted weapons use": 0,
            "Headquarters or base established": 0,
            "Looting/property destruction": 0,
            "Non-violent transfer of territory": 0,
            "Other": 0,
        }
        unmapped_events = set(df["sub_event_type"]) - set(acled_subevent_mapping.keys())

        if unmapped_events:
            raise ValueError(
                f"Unmapped sub_event_type(s) found in DataFrame: {unmapped_events}"
            )

        df["conflict"] = df["sub_event_type"].map(acled_subevent_mapping)

        return df

    def get_data(
        self,
        country: str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        *,
        event_types=None,
    ):
        """Gets data for a specified country and dates from the ACLED API.

        Args:
            country (str): Country to get data from.
            start_date (str | pd.Timestamp): Start date for pulling data. Accepts a
                YYYY-MM-DD string or a pd.Timestamp.
            end_date (str | pd.Timestamp): End date for pulling data. Accepts a
                YYYY-MM-DD string or a pd.Timestamp.
            event_types (list[str]): List of event types to pull data from. Event types must match ACLED event types.
        Returns:
            pd.DataFrame: Dataframe containing all data from the ACLED API.
        """
        self._validate_dates(start_date, end_date)
        self._validate_country(country)

        self.country = country
        self.start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        self.end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        self.event_types = event_types

        if not FORCE_DOWNLOAD:
            cached_df = self._read_data(country)
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
        mapped_df = self._mark_conflict_events(final_df)
        logger.info("All data successfully fetched.")
        self._save_data(mapped_df)
        return mapped_df
