import json

import streamlit as st

from ar_location_plant_map import PlantLocationFinder
from ui.chrome import definition_row, init_state, kicker, section_rule
from ui.presentation import extra_taxon_fields, photo_attribution, taxon_photo_url

init_state()
finder = PlantLocationFinder()

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(
    """<div class="pv-academic-badge">
        <span>📍</span>
        <span>Geospatial Biodiversity GIS & Observation Habitat Map</span>
    </div>""",
    unsafe_allow_html=True,
)
st.header("AR Field Explorer & Habitat Map")
st.markdown(
    "Perform spatial biodiversity analysis to discover verified wild and cultivated sightings "
    "of plant species in the vicinity of target coordinates. Spatial observations are indexed "
    "in real time from research-grade observation datasets curated by the global iNaturalist citizen science network."
)

st.space("small")
section_rule("section-rule-ar-controls")
st.space("small")

# ==============================================================================
# SPATIAL QUERY PARAMETERS
# ==============================================================================
with st.container(border=True):
    st.markdown("#### :material/tune: Spatial Query & Geospatial Coordinates")
    param_col1, param_col2 = st.columns([1.1, 0.9], gap="large")

    with param_col1:
        plant_name = st.text_input(
            "Target plant species (vernacular or botanical Latin)",
            value=st.session_state.get("ar_query") or "Neem",
            help="Specify common or binomial name, e.g. Neem, Azadirachta indica, Ginkgo, Oak, Rose",
        )

        st.caption("Suggested evaluation taxons:")
        quick_tags = st.container(horizontal=True, gap="small")
        with quick_tags:
            if st.button("🌿 Neem", key="ar_sug_neem"):
                plant_name = "Neem"
                st.session_state.ar_query = "Neem"
            if st.button("🍅 Tomato", key="ar_sug_tomato"):
                plant_name = "Tomato"
                st.session_state.ar_query = "Tomato"
            if st.button("🍏 Apple", key="ar_sug_apple"):
                plant_name = "Apple"
                st.session_state.ar_query = "Apple"
            if st.button("🥔 Potato", key="ar_sug_potato"):
                plant_name = "Potato"
                st.session_state.ar_query = "Potato"
            if st.button("🍃 Ginkgo", key="ar_sug_ginkgo"):
                plant_name = "Ginkgo"
                st.session_state.ar_query = "Ginkgo"
            if st.button("🌳 Oak", key="ar_sug_oak"):
                plant_name = "Oak"
                st.session_state.ar_query = "Oak"

        st.space("small")
        radius = st.slider(
            "Geographic search radius (kilometers)",
            min_value=1,
            max_value=100,
            value=st.session_state.get("ar_radius") or 25,
            help="Radius in kilometers around the target coordinate center.",
        )

    with param_col2:
        location_option = st.radio(
            "Coordinate acquisition method",
            ["Auto-detect my location", "Enter coordinates manually"],
            horizontal=True,
        )

        if location_option == "Auto-detect my location":
            detect = st.button("Detect Coordinates via IP", icon=":material/my_location:")
            if detect:
                with st.spinner("Acquiring IP geolocation coordinates..."):
                    location = finder.get_user_location()
                    if location:
                        st.session_state.ar_lat, st.session_state.ar_lon = location
                        st.success(
                            f"Location locked: {st.session_state.ar_lat:.4f}, {st.session_state.ar_lon:.4f}"
                        )
                    else:
                        st.error("Could not resolve IP geolocation. Please specify coordinates manually.")

            if st.session_state.get("ar_lat") is not None and st.session_state.get("ar_lon") is not None:
                st.caption(
                    f":material/check_circle: Locked coordinates: **{st.session_state.ar_lat:.4f}, {st.session_state.ar_lon:.4f}**"
                )
            user_lat = st.session_state.get("ar_lat")
            user_lon = st.session_state.get("ar_lon")
        else:
            c_lat, c_lon = st.columns(2)
            with c_lat:
                user_lat = st.number_input("Latitude", value=st.session_state.get("ar_lat") or 28.6139, format="%.6f")
            with c_lon:
                user_lon = st.number_input("Longitude", value=st.session_state.get("ar_lon") or 77.2090, format="%.6f")
            st.session_state.ar_lat = user_lat
            st.session_state.ar_lon = user_lon

    st.space("small")
    find = st.button("Execute Geospatial Sighting Query", type="primary", icon=":material/travel_explore:")

# ==============================================================================
# SPATIAL INFERENCE & INATURALIST QUERY
# ==============================================================================
if find:
    if not plant_name:
        st.error("Please specify a valid plant species name.")
        st.stop()
    if user_lat is None or user_lon is None:
        st.error("Please supply valid geographic coordinates.")
        st.stop()

    with st.status(f"Executing spatial biodiversity query for '{plant_name}'...", expanded=True) as status:
        st.write("1. Resolving species taxon ID in iNaturalist taxonomy registry...")
        plant_info = finder.search_plant_species(plant_name)
        if not plant_info:
            status.update(label="Species resolution failed", state="error")
            st.error(f"Could not resolve plant species '{plant_name}'. Please verify taxon spelling.")
            st.stop()

        st.write("2. Retrieving Wikipedia botanical overview...")
        scientific_name = plant_info.get("name", plant_name)
        description = finder.get_wikipedia_summary(scientific_name)

        st.write(f"3. Querying observations within geodesic radius of {radius} km...")
        observations = finder.get_plant_observations(
            plant_info["id"], user_lat, user_lon, radius
        )

        st.write("4. Computing geodesic distances (Haversine formula) and formatting spatial records...")
        if not observations:
            status.update(label="Query completed: zero nearby sightings recorded", state="complete")
            st.warning(f"No research-grade observations of '{plant_name}' recorded within {radius}km of target coordinates.")
            st.session_state.ar_results = []
            st.session_state.ar_plant_info = plant_info
            st.session_state.ar_description = description
            st.session_state.ar_query = plant_name
            st.session_state.ar_radius = radius
            st.stop()

        formatted_results = finder.format_observation_data(
            observations, user_lat, user_lon, plant_info, description
        )
        status.update(label=f"Query successful: found {len(formatted_results)} verified observations!", state="complete")
        st.session_state.ar_results = formatted_results
        st.session_state.ar_plant_info = plant_info
        st.session_state.ar_description = description
        st.session_state.ar_query = plant_name
        st.session_state.ar_radius = radius

formatted_results = st.session_state.get("ar_results")
plant_info = st.session_state.get("ar_plant_info")
description = st.session_state.get("ar_description")

if plant_info is None:
    st.stop()

scientific_name = plant_info.get("name", st.session_state.get("ar_query"))
common_name = plant_info.get("preferred_common_name", st.session_state.get("ar_query"))
photo_url = taxon_photo_url(plant_info)
credit = photo_attribution(plant_info)

st.space("large")
section_rule("section-rule-ar-results")
st.space("small")

# ==============================================================================
# SPECIES TAXONOMIC PROFILE
# ==============================================================================
kicker("Species Profile", key="kicker-ar-profile")

with st.container(border=True):
    visual_col, identity_col = st.columns([0.85, 1.15], gap="large", vertical_alignment="center")

    with visual_col:
        with st.container(key="library-image"):
            if photo_url:
                st.image(photo_url, alt=f"Photograph of {common_name}", width="stretch")
                if credit:
                    st.caption(credit)
            else:
                st.image(
                    "static/ginkgo.jpg",
                    alt="Botanical reference photograph",
                    width="stretch",
                )
                st.caption("Botanical reference photograph • Wikimedia Commons")

    with identity_col:
        rank = plant_info.get("rank")
        if rank:
            st.markdown(f":green-badge[{str(rank).replace('_', ' ').upper()}]")
        st.markdown(f"## {common_name or 'Unknown'}")
        st.markdown(f"### *{scientific_name}*")

        st.space("small")
        with st.container(border=True):
            definition_row("Common Name", str(common_name or "Unknown"), key="ar-common")
            definition_row("Scientific Binomial", f"*{scientific_name}*", key="ar-sci")
            if rank:
                definition_row("Taxonomic Rank", str(rank).replace("_", " ").title(), key="ar-rank")

st.space("medium")

with st.expander(":material/description: Botanical Encyclopedia Abstract", expanded=False):
    st.markdown(description or "No encyclopedic summary available from Wikipedia.")

# ==============================================================================
# INTERACTIVE MAP & SIGHTINGS LOG
# ==============================================================================
if formatted_results:
    st.space("medium")
    kicker("Geographic Distribution", key="kicker-ar-map")
    st.subheader(f"Interactive Observation Map ({len(formatted_results)} Records)")
    st.caption(
        f"Georeferenced field observations of **{common_name}** within {st.session_state.get('ar_radius', 25)} km "
        f"of coordinates ({st.session_state.get('ar_lat', 0):.4f}, {st.session_state.get('ar_lon', 0):.4f})."
    )

    # Compile coordinate points for interactive map
    map_points = [
        {"latitude": r["latitude"], "longitude": r["longitude"]}
        for r in formatted_results
        if r.get("latitude") and r.get("longitude")
    ]
    if map_points:
        st.map(map_points, zoom=9)

    st.space("large")
    kicker("Field Voucher Records", key="kicker-ar-nearby")
    st.subheader("Observation Log & Vouchers")

    for index, result in enumerate(formatted_results[:12]):
        with st.container(border=True):
            info_col, photo_col = st.columns([1.2, 0.8], gap="large", vertical_alignment="center")

            with info_col:
                st.markdown(
                    f":blue-badge[Geodesic Distance: {result['distance_km']} km] "
                    f":green-badge[{result['quality_grade'].replace('_', ' ').title()}]"
                )
                st.markdown(f"#### {result['location_name']}")
                st.caption(
                    f"Coordinates: **{result['latitude']:.4f}, {result['longitude']:.4f}** • "
                    f"Observation Date: **{result['observed_on']}**"
                )

                if result["url"]:
                    st.markdown(f"[:material/open_in_new: View primary record on iNaturalist]({result['url']})")

            with photo_col:
                if result.get("photos"):
                    for obs_photo in result["photos"]:
                        try:
                            st.image(
                                obs_photo,
                                width="stretch",
                                alt="iNaturalist observation photograph",
                            )
                        except Exception:
                            st.caption("Photo preview unavailable")
                else:
                    st.caption("No field photograph attached")

    json_data = {
        "search_query": st.session_state.get("ar_query"),
        "user_location": {
            "latitude": st.session_state.get("ar_lat"),
            "longitude": st.session_state.get("ar_lon"),
        },
        "search_radius_km": st.session_state.get("ar_radius"),
        "plant_info": plant_info,
        "results": formatted_results,
    }
    st.space("medium")
    st.download_button(
        label="Export Observation GIS Dataset (JSON)",
        data=json.dumps(json_data, indent=2),
        file_name=f"plant_locations_{(st.session_state.get('ar_query') or 'plant').replace(' ', '_')}.json",
        mime="application/json",
        icon=":material/download:",
    )

# ==============================================================================
# LOCAL LOGGED PATHOLOGY SIGHTINGS GIS LAYER
# ==============================================================================
local_sightings = [
    d for d in st.session_state.get("disease_diagnoses", [])
    if d.get("is_sighting") and d.get("latitude") is not None and d.get("longitude") is not None
]

if local_sightings:
    st.space("large")
    section_rule("section-rule-local-sightings")
    st.space("small")
    kicker("Local Researcher Network", key="kicker-local-sightings")
    st.subheader(f"Logged Field Pathology Sightings ({len(local_sightings)} Records)")
    st.caption("Locally cataloged diseased and healthy specimen observations anchored to geographic coordinates.")

    loc_points = [
        {"latitude": float(s["latitude"]), "longitude": float(s["longitude"])}
        for s in local_sightings
    ]
    if loc_points:
        st.map(loc_points, zoom=8)

    st.space("medium")
    for s_idx, s in enumerate(local_sightings[:6]):
        with st.container(border=True):
            s_col1, s_col2 = st.columns([1.2, 0.8], gap="large", vertical_alignment="center")
            with s_col1:
                status_str = s.get("health_status", "Diseased")
                if status_str == "Healthy":
                    st.markdown(":green-badge[✓ Healthy Foliage Sighting]")
                else:
                    st.markdown(f":red-badge[⚠ Pathological Outbreak: {s.get('disease_name', 'Condition')}]")

                st.markdown(f"#### {s.get('plant_name', 'Host')} — {s.get('location_name', 'Field Location')}")
                st.caption(
                    f"Coordinates: **{float(s['latitude']):.4f}, {float(s['longitude']):.4f}** • "
                    f"Logged on: **{s.get('created_at', 'Session')}**"
                )
                if s.get("notes"):
                    st.markdown(f":material/notes: *\"{s['notes']}\"*")
                if s.get("pathogen"):
                    st.caption(f"Identified Pathogen: *{s['pathogen']}* (Confidence: {s.get('confidence_pct', 0):.1f}%)")

            with s_col2:
                if s.get("image_bytes"):
                    st.image(
                        s["image_bytes"],
                        alt=s.get("disease_name") or "Field sighting",
                        width="stretch",
                    )
                else:
                    st.caption("No photographic voucher attached")

