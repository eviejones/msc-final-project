import os
import pandas as pd
import requests
import json
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Acled():
    def __init__(self):
        load_dotenv()
        self.endpoint = "https://acleddata.com/api/acled/read?_format=json"
        self.username = os.getenv("ACLED_USERNAME")
        self.password = os.getenv("ACLED_PASSWORD")
        self.token_url = os.getenv("ACLED_TOKEN_URL")
        self.access_token = self._get_access_token()

    def _get_access_token(self):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
            "client_id": "acled",
        }
        response = requests.post(
            self.token_url, headers=headers, data=data, timeout=20
        )

        if response.status_code == 200:
            token_data = response.json()
            logger.info("Access token correctly retrieved.")
            return token_data["access_token"]
        raise Exception(
            f"Failed to get access token: {response.status_code} {response.text}"
        )

    def _build_params(self) -> dict:
        if isinstance(self.countries, str):
            self.countries = [self.countries]

        if self.event_types is not None and isinstance(self.event_types, str):
            self.event_types = [self.event_types]

        params = {
            "country": ":OR:country=".join(self.countries),
            "event_date": f"{self.start_date}|{self.end_date}",
            "event_date_where": "BETWEEN",
        }
        if self.event_types is not None:
            params["event_type"] = ":OR:event_type=".join(self.event_types)
        return params

    def get_data(self, countries: list[str], start_date: str, end_date: str, event_types=None):
        self.countries = countries
        self.start_date = start_date
        self.end_date = end_date
        self.event_types = event_types
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

                if len(r.json()["data"]) < 5000:
                    request_end = True
                else:
                    params["page"] += 1
            else:
                raise requests.RequestException(
                    f"HTTP Code: {r.status_code}, Status: {r.reason}"
                )
        final_df = pd.concat(r_dfs)
        final_df["event_date"] = pd.to_datetime(final_df["event_date"])
        logger.info("All data successfully fetched.")
        return final_df