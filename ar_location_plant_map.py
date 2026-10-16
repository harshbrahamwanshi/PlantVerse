import math
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st


class PlantLocationFinder:
    def __init__(self):
        self.inaturalist_base_url = "https://api.inaturalist.org/v1"
        self.wikipedia_base_url = "https://en.wikipedia.org/api/rest_v1"
        self.ipinfo_url = "http://ip-api.com/json"

    def get_user_location(self) -> Optional[Tuple[float, float]]:
        """Get user's approximate location using IP geolocation from ip-api.com."""
        try:
            response = requests.get(self.ipinfo_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                lat = data.get("lat")
                lon = data.get("lon")
                if lat is not None and lon is not None:
                    return lat, lon
                else:
                    st.error("Could not extract lat/lon from response.")
            else:
                st.error(f"Geolocation request failed with status code: {response.status_code}")
        except Exception as e:
            st.error(f"Could not auto-detect location: {e}")
        return None

    def search_plant_species(self, plant_name: str) -> Optional[Dict]:
        """Search for plant species in iNaturalist to get taxon ID."""
        try:
            url = f"{self.inaturalist_base_url}/taxa"
            params = {
                "q": plant_name,
                "rank": "species",
                "is_active": "true",
                "per_page": 1,
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data["results"]:
                    return data["results"][0]
        except Exception as e:
            st.error(f"Error searching for plant species: {e}")
        return None

    def get_plant_observations(
        self, taxon_id: int, lat: float, lon: float, radius: int = 50
    ) -> List[Dict]:
        """Get plant observations from iNaturalist near the specified location."""
        try:
            url = f"{self.inaturalist_base_url}/observations"
            params = {
                "taxon_id": taxon_id,
                "lat": lat,
                "lng": lon,
                "radius": radius,
                "per_page": 50,
                "order": "desc",
                "order_by": "observed_on",
                "quality_grade": "research,needs_id",
                "photos": "true",
                "geo": "true",
            }

            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
        except Exception as e:
            st.error(f"Error fetching observations: {e}")
        return []

    def get_wikipedia_summary(self, scientific_name: str) -> str:
        """Get plant description from Wikipedia."""
        try:
            search_term = scientific_name.replace(" ", "_")
            url = f"{self.wikipedia_base_url}/page/summary/{urllib.parse.quote(search_term)}"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("extract", "No description available.")
            else:
                search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
                alt_search = scientific_name.split()[0]
                response = requests.get(
                    f"{search_url}{urllib.parse.quote(alt_search)}", timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("extract", "No description available.")
        except Exception:
            pass

        return "Description not available from Wikipedia."

    def calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two coordinates in kilometers."""
        R = 6371

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def format_observation_data(
        self,
        observations: List[Dict],
        user_lat: float,
        user_lon: float,
        plant_info: Dict,
        description: str,
    ) -> List[Dict]:
        """Format observation data for display."""
        formatted_data = []

        for obs in observations:
            if not obs.get("geojson") or not obs["geojson"].get("coordinates"):
                continue

            coords = obs["geojson"]["coordinates"]
            obs_lon, obs_lat = coords[0], coords[1]

            distance = self.calculate_distance(user_lat, user_lon, obs_lat, obs_lon)

            location_name = obs.get(
                "place_guess", f"Location ({obs_lat:.4f}, {obs_lon:.4f})"
            )

            formatted_obs = {
                "location_name": location_name,
                "latitude": obs_lat,
                "longitude": obs_lon,
                "distance_km": round(distance, 2),
                "observed_on": obs.get("observed_on_string", "Unknown date"),
                "quality_grade": obs.get("quality_grade", "unknown"),
                "url": obs.get("uri", ""),
                "photos": [photo["url"] for photo in obs.get("photos", [])[:2]],
                "description": description,
                "scientific_name": plant_info.get("name", "Unknown"),
                "common_name": plant_info.get("preferred_common_name", "Unknown"),
            }

            formatted_data.append(formatted_obs)

        formatted_data.sort(key=lambda x: x["distance_km"])
        return formatted_data
