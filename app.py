"""
╔══════════════════════════════════════════════════════════════════════╗
║   Hong Kong Recycling Accessibility — Hackathon Presentation        ║
║   Real EPD datasets · 8,796 collection points · 2009-2024 stats     ║
╚══════════════════════════════════════════════════════════════════════╝

RUN:
    pip install streamlit folium streamlit-folium pandas numpy scipy networkx
    streamlit run app.py

PUT THESE FILES IN THE SAME FOLDER as app.py:
    清洗4_1_.csv
    clean_1回收站开放空间数据库.geojson
    clean_2可回收收集点数据.csv
    clean_3废物管理设施.csv
    数据库7.csv
    数据库5.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import json, math, os, heapq
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
def dp(n): return os.path.join(DATA_DIR, n)

# ── page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HK Recycling Accessibility",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0f0a; color: #e8f0e8; }
[data-testid="stSidebar"] { background: #0f1a0f; border-right: 1px solid #1e3a1e; }
div[data-testid="metric-container"] {
    background: #111e11; border: 1px solid #1e3a1e; border-radius: 10px; padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    color: #5a8f5a !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.1em; font-family: 'DM Mono', monospace !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #d4f0d4 !important; font-size: 1.9rem !important; font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #5a8f5a !important; }
h1 { color: #d4f0d4 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2 { color: #5a8f5a !important; font-size: 0.78rem !important; text-transform: uppercase;
     letter-spacing: 0.12em; border-bottom: 1px solid #1e3a1e; padding-bottom: 6px; margin-top: 20px !important; }
h3 { color: #90c890 !important; font-weight: 500 !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #0f1a0f; border-radius: 8px; border: 1px solid #1e3a1e; padding: 3px; gap: 3px;
}
.stTabs [data-baseweb="tab"] { background: transparent; color: #5a8f5a; border-radius: 5px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #1e3a1e !important; color: #90c890 !important; }
.chip {
    display: inline-block; background: #1e3a1e; color: #90c890;
    border: 1px solid #2d5a2d; border-radius: 20px;
    padding: 2px 10px; font-size: 0.73rem; margin: 2px;
    font-family: 'DM Mono', monospace;
}
.fact-card {
    background: #111e11; border: 1px solid #1e3a1e; border-left: 3px solid #3fb950;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-size: 0.85rem; color: #8ab88a;
}
.warn-card {
    background: #1a1100; border: 1px solid #3d2b00; border-left: 3px solid #d29922;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-size: 0.84rem; color: #c8a840;
}
</style>
""", unsafe_allow_html=True)

HK_CENTER = [22.3193, 114.1694]

# ══════════════════════════════════════════════════════════════════════
# SPATIAL UTIL
# ══════════════════════════════════════════════════════════════════════
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ══════════════════════════════════════════════════════════════════════
# SIMPLIFIED MATERIAL CATEGORY SYSTEM
# ══════════════════════════════════════════════════════════════════════
# Consolidate 16 raw waste types → 5 meaningful categories.
# This reduces visual clutter while preserving the key policy-relevant
# distinctions: paper, plastics, metals, e-waste, and others.

MATERIAL_CATEGORY = {
    "Paper":                                     ("Paper",     "#79c0ff"),
    "Plastics":                                  ("Plastics",  "#d2a8ff"),
    "Plastic Bottle":                            ("Plastics",  "#d2a8ff"),
    "Beverage Cartons":                          ("Plastics",  "#d2a8ff"),
    "Metals":                                    ("Metals",    "#ffa657"),
    "Glass Bottles":                             ("Metals",    "#ffa657"),
    "Computers":                                 ("E-waste",   "#58a6ff"),
    "Small Electrical and Electronic Equipment": ("E-waste",   "#58a6ff"),
    "Regulated Electrical Equipment":            ("E-waste",   "#58a6ff"),
    "Rechargeable Batteries":                    ("E-waste",   "#58a6ff"),
    "Fluorescent Lamp":                          ("E-waste",   "#58a6ff"),
    "Printer Cartridges":                        ("E-waste",   "#58a6ff"),
    "Clothes":                                   ("Others",    "#f0883e"),
    "Food Waste":                                ("Others",    "#f0883e"),
    "Barbeque Fork":                             ("Others",    "#f0883e"),
    "Other Description":                         ("Others",    "#f0883e"),
}
# 5 canonical categories with display colour
CATEGORY_COLOR = {
    "Paper":    "#79c0ff",
    "Plastics": "#d2a8ff",
    "Metals":   "#ffa657",
    "E-waste":  "#58a6ff",
    "Others":   "#f0883e",
}

# Station-tier adaptive walking-radius (metres) — service design reference.
# Used as FALLBACK when OSMnx walk isochrone is unavailable.
STATION_WALK_RADIUS = {
    "GREEN@ Hub":                            2000,
    "Recycling Stations/Recycling Stores":    800,
    "NGO Collection Points":                  600,
    "Smart Bin":                              400,
    "Private Collection Points (e.g. housing estates, shopping centres)": 250,
    "Recycling Bins at Public Place":         200,
    "Recycling Spots":                        200,
    "Street Corner Recycling Shops":          300,
}
STATION_WALK_RADIUS_DEFAULT = 200

def get_station_radius(legend: str) -> int:
    return STATION_WALK_RADIUS.get(legend, STATION_WALK_RADIUS_DEFAULT)

def parse_materials(waste_type_str) -> list:
    if pd.isna(waste_type_str) or not str(waste_type_str).strip():
        return []
    return [m.strip() for m in str(waste_type_str).split(",") if m.strip()]

def materials_to_categories(waste_type_str) -> list:
    """Return deduplicated list of (category, colour) for a waste_type string."""
    seen = {}
    for mat in parse_materials(waste_type_str):
        cat, col = MATERIAL_CATEGORY.get(mat, ("Others", "#f0883e"))
        seen[cat] = col
    return list(seen.items())


# ══════════════════════════════════════════════════════════════════════
# WALK NETWORK — OSMnx pedestrian graph + isochrone utilities
# ══════════════════════════════════════════════════════════════════════
WALK_CACHE = dp("hk_walk_graph.graphml")

@st.cache_resource
def build_walk_network_osmnx():
    """
    Load (or download once) the HK pedestrian network via OSMnx.
    network_type='walk' includes footways, paths, and streets walkable
    on foot — far more accurate than a circuity multiplier on haversine.

    Returns lightweight structures identical to build_road_network_osmnx:
        walk_adj    : dict  osmid → [(neighbour, metres)]
        walk_coords : dict  osmid → (lat, lon)
        walk_grid   : dict  cell  → [osmid …]
        GRID        : float
    """
    try:
        import osmnx as ox
    except ImportError:
        return None, None, None, None

    GRID = 0.01
    try:
        if os.path.exists(WALK_CACHE):
            G = ox.load_graphml(WALK_CACHE)
        else:
            with st.spinner("Downloading HK walk network (one-time ~20 s)…"):
                G = ox.graph_from_place("Hong Kong",
                                        network_type="walk",
                                        simplify=True)
                ox.save_graphml(G, WALK_CACHE)
    except Exception:
        return None, None, None, None

    walk_adj    = defaultdict(list)
    walk_coords = {}
    for u, v, data in G.edges(data=True):
        w = data.get("length", 1.0)
        walk_adj[u].append((v, w))
        walk_adj[v].append((u, w))
    for node, data in G.nodes(data=True):
        walk_coords[node] = (data["y"], data["x"])

    walk_grid = defaultdict(list)
    for osmid, (lat, lon) in walk_coords.items():
        cell = (round(lon // GRID * GRID, 3), round(lat // GRID * GRID, 3))
        walk_grid[cell].append(osmid)

    return dict(walk_adj), walk_coords, dict(walk_grid), GRID


def _snap_walk(lat, lon, walk_grid, walk_coords, GRID, r=5):
    """Snap (lat,lon) to nearest walk-graph node."""
    cx = round(lon // GRID * GRID, 3)
    cy = round(lat // GRID * GRID, 3)
    best_d, best_n = float("inf"), None
    for ri in range(1, r + 1):
        for dx in range(-ri, ri + 1):
            for dy in range(-ri, ri + 1):
                cell = (round(cx + dx * GRID, 3), round(cy + dy * GRID, 3))
                for n in walk_grid.get(cell, []):
                    nlat, nlon = walk_coords[n]
                    d = haversine_m(lat, lon, nlat, nlon)
                    if d < best_d:
                        best_d, best_n = d, n
        if best_n and best_d < 800:
            break
    return best_n


def walk_dist_to_nearest(origin_lat, origin_lon,
                          target_coords,
                          walk_adj, walk_coords, walk_grid, GRID,
                          max_dist_m=5000):
    """
    True pedestrian network distance from origin to the nearest point
    in target_coords, using Dijkstra on the OSMnx walk graph.

    Falls back to haversine × 1.3 circuity if the walk graph is
    unavailable or the path exceeds max_dist_m.
    """
    if walk_adj is None:
        straight = min(haversine_m(origin_lat, origin_lon, t[0], t[1])
                       for t in target_coords)
        return straight * 1.3

    src = _snap_walk(origin_lat, origin_lon, walk_grid, walk_coords, GRID)
    if src is None:
        straight = min(haversine_m(origin_lat, origin_lon, t[0], t[1])
                       for t in target_coords)
        return straight * 1.3

    # Snap all targets
    tgt_nodes = set()
    for tlat, tlon in target_coords:
        n = _snap_walk(tlat, tlon, walk_grid, walk_coords, GRID)
        if n is not None:
            tgt_nodes.add(n)

    if not tgt_nodes:
        straight = min(haversine_m(origin_lat, origin_lon, t[0], t[1])
                       for t in target_coords)
        return straight * 1.3

    # Multi-target Dijkstra (stop at first target reached)
    dist_map = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in tgt_nodes:
            return d
        if d > dist_map.get(u, float("inf")) or d > max_dist_m:
            continue
        for v, w in walk_adj.get(u, []):
            nd = d + w
            if nd < dist_map.get(v, float("inf")):
                dist_map[v] = nd
                heapq.heappush(pq, (nd, v))

    # Unreachable within budget — fall back
    straight = min(haversine_m(origin_lat, origin_lon, t[0], t[1])
                   for t in target_coords)
    return straight * 1.3


def walk_isochrone_polygon(origin_lat, origin_lon, radius_m,
                            walk_adj, walk_coords, walk_grid, GRID,
                            n_spokes=24):
    """
    Compute a convex-ish walk isochrone polygon for Folium by:
      1. Running Dijkstra from origin up to radius_m on the walk graph.
      2. Collecting all reachable nodes.
      3. Taking the convex hull via a spoke-sampling approach
         (angles 0°→360°, furthest reachable node per spoke).

    Returns list of [lat, lon] pairs forming a closed polygon, or None
    if the walk graph is unavailable (caller falls back to a circle).
    """
    if walk_adj is None:
        return None

    src = _snap_walk(origin_lat, origin_lon, walk_grid, walk_coords, GRID)
    if src is None:
        return None

    # Dijkstra limited to radius_m
    dist_map = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist_map.get(u, float("inf")) or d > radius_m:
            continue
        for v, w in walk_adj.get(u, []):
            nd = d + w
            if nd < dist_map.get(v, float("inf")) and nd <= radius_m:
                dist_map[v] = nd
                heapq.heappush(pq, (nd, v))

    if len(dist_map) < 4:
        return None

    # Spoke-based boundary: divide 360° into n_spokes, pick furthest node
    spoke_pts = {}
    for node, d in dist_map.items():
        nlat, nlon = walk_coords[node]
        angle = math.degrees(math.atan2(nlon - origin_lon, nlat - origin_lat)) % 360
        spoke = int(angle / (360 / n_spokes))
        if spoke not in spoke_pts or d > spoke_pts[spoke][2]:
            spoke_pts[spoke] = (nlat, nlon, d)

    if len(spoke_pts) < 4:
        return None

    # Sort by spoke index to form a closed polygon
    boundary = [spoke_pts[k] for k in sorted(spoke_pts.keys())]
    poly = [[pt[0], pt[1]] for pt in boundary]
    poly.append(poly[0])   # close
    return poly

# ══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_recycling_points():
    """
    8,796 real EPD collection points.
    Classification: Smart Bins + Stations → "premium"; rest → "basic"
    """
    df = pd.read_csv(dp("recycling_points.csv"))
    df = df.rename(columns={"lgt": "lon"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    premium = {"Smart Bin", "Recycling Stations/Recycling Stores", "NGO Collection Points"}
    df["point_type"] = df["legend"].apply(lambda x: "premium" if x in premium else "basic")
    return df


@st.cache_data
def load_green_stations():
    """
    12 GREEN@ flagship recycling hubs.
    Coordinates extracted as polygon centroids from GeoJSON.
    UNDERSERVED PROXY: distance to nearest GREEN@ hub (not just a bin)
    captures the gap in premium/convenient recycling access.
    """
    with open(dp("green_stations.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    rows = []
    for feat in gj["features"]:
        props = feat["properties"]
        geom  = feat["geometry"]
        coords = []
        if geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly: coords.extend(ring)
        elif geom["type"] == "Polygon":
            for ring in geom["coordinates"]: coords.extend(ring)
        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            rows.append({
                "name":    props.get("bldg_engnm", "GREEN Station"),
                "name_zh": props.get("bldg_chtnm", ""),
                "lat":     sum(lats) / len(lats),
                "lon":     sum(lons) / len(lons),
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_housing_estates():
    """240 HK housing estates with GPS coordinates and flat counts."""
    df = pd.read_csv(dp("housing_estates.csv"))
    df["lat"]         = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"]         = pd.to_numeric(df["lng"], errors="coerce")
    df["no_of_flats"] = pd.to_numeric(df["no_of_flats"], errors="coerce").fillna(0)
    return df.dropna(subset=["lat", "lng"])


@st.cache_data
def load_cost_data():
    """Load operational cost parameters from recycling_cost.json."""
    path = dp("recycling_cost.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_waste_stats():
    """EPD annual MSW generation, recovery and disposal 2009-2024."""
    df  = pd.read_csv(dp("stats.csv"), low_memory=False)
    gen = df[df["source_file"] == "市政固体废物产生量.csv"][["year", "generation_q"]]
    rec = df[df["source_file"] == "市政固体废物回收量.csv"][["year", "recovery_q"]]
    dis = df[(df["source_file"] == "按废物类别划分的固体废物总处置量.csv")
             & (df["waste_cat_sc"] == "都市固体废物")][["year", "disposal_q"]]
    merged = gen.merge(rec, on="year").merge(dis, on="year")
    merged["recycling_rate"] = merged["recovery_q"] / merged["generation_q"] * 100
    merged["year"] = merged["year"].astype(int)
    return merged.sort_values("year")


# ══════════════════════════════════════════════════════════════════════
# PRE-COMPUTE DISTANCES (cached once at startup)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data
def compute_distances_to_green(estate_lats, estate_lngs, green_lats, green_lons):
    """
    Haversine straight-line distances to nearest GREEN@ hub.
    Used as the raw input to compute_walk_distances_to_green when the
    walk graph is unavailable, and for display tooltips.
    """
    dists = []
    green_coords = list(zip(green_lats, green_lons))
    for lat, lng in zip(estate_lats, estate_lngs):
        d = min(haversine_m(lat, lng, g[0], g[1]) for g in green_coords)
        dists.append(d)
    return np.array(dists)


@st.cache_data
def compute_walk_distances_to_green(estate_lats, estate_lngs,
                                     green_lats, green_lons,
                                     _walk_adj, _walk_coords, _walk_grid, _GRID):
    """
    True OSMnx pedestrian-network distances from each housing estate to
    the nearest GREEN@ hub.  Falls back to haversine×1.3 if the walk
    graph is unavailable.

    Results are cached per (estate coords, green coords) tuple so
    Streamlit does not recompute on every render.
    """
    green_coords = list(zip(green_lats, green_lons))
    dists = []
    for lat, lng in zip(estate_lats, estate_lngs):
        d = walk_dist_to_nearest(lat, lng, green_coords,
                                  _walk_adj, _walk_coords, _walk_grid, _GRID)
        dists.append(d)
    return np.array(dists)


@st.cache_data
def compute_distances_to_rcp(estate_lats, estate_lngs, rcp_lats, rcp_lons):
    """Straight-line distance from each estate to nearest basic collection point."""
    dists = []
    r_coords = list(zip(rcp_lats, rcp_lons))
    for lat, lng in zip(estate_lats, estate_lngs):
        d = min(haversine_m(lat, lng, r[0], r[1]) for r in r_coords)
        dists.append(d)
    return np.array(dists)


@st.cache_data
def compute_distances_to_small_stations(estate_lats, estate_lngs, rcp_lats, rcp_lons):
    """Straight-line distance to nearest small/medium premium station."""
    r_coords = list(zip(rcp_lats, rcp_lons))
    if not r_coords:
        return np.full(len(estate_lats), 1e9)
    dists = []
    for lat, lng in zip(estate_lats, estate_lngs):
        d = min(haversine_m(lat, lng, r[0], r[1]) for r in r_coords)
        dists.append(d)
    return np.array(dists)


@st.cache_data
def compute_walk_distances_to_small(estate_lats, estate_lngs,
                                     small_lats, small_lons,
                                     _walk_adj, _walk_coords, _walk_grid, _GRID):
    """True pedestrian-network distance to nearest small/medium station."""
    if not small_lats:
        return np.full(len(estate_lats), 1e9)
    small_coords = list(zip(small_lats, small_lons))
    dists = []
    for lat, lng in zip(estate_lats, estate_lngs):
        d = walk_dist_to_nearest(lat, lng, small_coords,
                                  _walk_adj, _walk_coords, _walk_grid, _GRID)
        dists.append(d)
    return np.array(dists)


def compute_composite_accessibility(
    green_dists: np.ndarray,
    small_dists: np.ndarray,
    green_weight: float = 1.0,
    small_weight: float = 0.5,
) -> np.ndarray:
    """
    Composite pedestrian-accessibility effective distance.
    All inputs should be walk-network distances.

    effective_dist = min(walk_green / 1.0,  walk_small / 0.5)

    A small station at 1 km walk = effective 2 km  (borderline served)
    A GREEN@ hub   at 2 km walk = effective 2 km  (just served)
    """
    return np.minimum(green_dists / green_weight, small_dists / small_weight)


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# MAP BUILDERS
# ══════════════════════════════════════════════════════════════════════

def build_main_map(rcp_df, green_df, estates_df,
                   service_radius_m, show_heat, show_rcp, show_green,
                   show_coverage, show_underserved, rcp_filter,
                   green_dists, composite_dists=None,
                   walk_adj=None, walk_coords=None,
                   walk_grid=None, walk_GRID=None):
    """
    Accessibility map with:
    1. SIMPLIFIED 5-CATEGORY material colours (Paper/Plastics/Metals/E-waste/Others)
    2. OSMnx walk-isochrone polygons for GREEN@ hubs and premium stations —
       true pedestrian reachability instead of uniform circles.
       Falls back to a styled circle if walk graph unavailable.
    3. Composite walk-distance underserved detection.
    """
    m = folium.Map(location=HK_CENTER, zoom_start=11,
                   tiles="CartoDB dark_matter", control_scale=True)

    # ── Layer 1: Population density heatmap ──────────────────────────
    # Lighter rendering: lower max intensity, softer gradient, smaller radius
    if show_heat:
        heat_data = [[r["lat"], r["lng"], min(float(r["no_of_flats"]) / 3500.0, 0.65)]
                     for _, r in estates_df.iterrows()]
        HeatMap(heat_data, radius=20, blur=30, max_zoom=13,
                gradient={"0.0": "transparent",
                           "0.3": "rgba(26,74,26,0.30)",
                           "0.6": "rgba(255,238,0,0.35)",
                           "0.85": "rgba(255,102,0,0.28)",
                           "1.0": "rgba(255,0,0,0.22)"}).add_to(m)

    # Filter by type
    filt = rcp_df if rcp_filter == "All" else rcp_df[rcp_df["point_type"] == rcp_filter.lower()]

    # ── Layer 2: Material-category coverage zones ────────────────────
    # All premium stations (GREEN@ + smaller) get OSMnx walk-isochrone
    # polygons when the walk graph is available — true pedestrian catchment
    # rather than uniform circles.  Basic bins use brighter circles
    # (visible but no isochrone — too numerous for Dijkstra expansion).
    if show_coverage:
        # --- GREEN@ walk isochrones -------------------------------------------
        green_iso_grp = folium.FeatureGroup(name="GREEN@ Walk Isochrones", show=True)
        GREEN_RADIUS = STATION_WALK_RADIUS["GREEN@ Hub"]
        for _, r in green_df.iterrows():
            poly = walk_isochrone_polygon(
                r["lat"], r["lon"], GREEN_RADIUS,
                walk_adj, walk_coords, walk_grid, walk_GRID
            ) if walk_adj else None
            if poly:
                folium.Polygon(
                    poly,
                    color="#3fb950", fill=True, fill_color="#3fb950",
                    fill_opacity=0.14, weight=2.2, opacity=0.85,
                    dash_array="6 3",
                    tooltip=f"GREEN@ walk catchment ({GREEN_RADIUS}m) — {r['name']}",
                ).add_to(green_iso_grp)
            else:
                folium.Circle(
                    [r["lat"], r["lon"]], radius=GREEN_RADIUS,
                    color="#3fb950", fill=True, fill_color="#3fb950",
                    fill_opacity=0.10, weight=2.0, dash_array="6 3",
                    tooltip=f"GREEN@ radius {GREEN_RADIUS}m — {r['name']}",
                ).add_to(green_iso_grp)
        green_iso_grp.add_to(m)

        # --- Premium small stations: walk isochrone OR coloured circle --------
        # Stations with walk graph: real isochrone polygon per category colour.
        # Without walk graph: brighter circles (fill_opacity raised significantly).
        prem_cov = folium.FeatureGroup(name="Station Coverage (by material)", show=True)
        prem_pts = filt[filt["point_type"] == "premium"]
        # Legends that get walk isochrone treatment (worth the Dijkstra cost)
        ISO_LEGENDS = {"Recycling Stations/Recycling Stores", "NGO Collection Points", "Smart Bin"}
        for _, r in prem_pts.iloc[::2].iterrows():
            legend_val = r.get("legend", "")
            if legend_val in ("GREEN@ Hub",):
                continue
            radius = get_station_radius(legend_val)
            cats   = materials_to_categories(r.get("waste_type", ""))
            # Pick the dominant colour: first category or neutral
            dom_col = cats[0][1] if cats else "#5a8f5a"

            if walk_adj and legend_val in ISO_LEGENDS:
                # True walk isochrone — single polygon in dominant material colour
                poly = walk_isochrone_polygon(
                    r["lat"], r["lon"], radius,
                    walk_adj, walk_coords, walk_grid, walk_GRID
                )
                if poly:
                    cat_label = "/".join(c for c, _ in cats) if cats else legend_val
                    folium.Polygon(
                        poly,
                        color=dom_col, fill=True, fill_color=dom_col,
                        fill_opacity=0.16, weight=1.6, opacity=0.75,
                        tooltip=f"{cat_label} · {legend_val} · {radius}m walk",
                    ).add_to(prem_cov)
                    continue   # polygon drawn — skip circle fallback

            # Circle fallback (no walk graph, or non-ISO legend, or poly failed)
            if not cats:
                folium.Circle([r["lat"], r["lon"]], radius=radius,
                              color="#5a8f5a", fill=True, fill_opacity=0.16,
                              weight=1.4, opacity=0.65).add_to(prem_cov)
            else:
                for cat, col in cats:
                    folium.Circle([r["lat"], r["lon"]], radius=radius,
                                  color=col, fill=True, fill_opacity=0.16,
                                  weight=1.2, opacity=0.60,
                                  tooltip=f"{cat} · {legend_val} · {radius}m walk",
                                  ).add_to(prem_cov)
        prem_cov.add_to(m)

        # --- Basic bins: brighter circles, default ON, sampled 1:6 -----------
        basic_cov = folium.FeatureGroup(name="Basic Bin Coverage", show=True)
        basic_pts = filt[filt["point_type"] == "basic"]
        for _, r in basic_pts.iloc[::6].iterrows():
            cats   = materials_to_categories(r.get("waste_type", ""))
            b_col  = cats[0][1] if cats else "#6a9abf"
            folium.Circle([r["lat"], r["lon"]], radius=200,
                          color=b_col, fill=True, fill_opacity=0.13,
                          weight=1.0, opacity=0.55,
                          tooltip=f"{cats[0][0] if cats else 'Mixed'} · public bin · 200m",
                          ).add_to(basic_cov)
        basic_cov.add_to(m)

    # ── Layer 3: Clustered collection point markers ───────────────────
    if show_rcp:
        rcp_grp = folium.FeatureGroup(name="Collection Points", show=True)
        cluster = MarkerCluster(options={"maxClusterRadius": 40,
                                          "disableClusteringAtZoom": 15})
        for _, r in filt.iterrows():
            legend_val = r.get("legend", "")
            cats = materials_to_categories(r.get("waste_type", ""))
            radius_m = get_station_radius(legend_val)
            # Colour by tier
            if legend_val == "Smart Bin":
                mcolor, micon = "blue", "bolt"
            elif legend_val in ("Recycling Stations/Recycling Stores",
                                "NGO Collection Points"):
                mcolor, micon = "green", "recycle"
            elif r["point_type"] == "premium":
                mcolor, micon = "darkgreen", "recycle"
            else:
                mcolor, micon = "lightgray", "trash"
            # Category badge HTML
            cat_dots = " ".join(
                f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                f'background:{col};margin:0 2px;vertical-align:middle" title="{cat}"></span>'
                for cat, col in cats
            ) or "—"
            folium.Marker(
                [r["lat"], r["lon"]],
                icon=folium.Icon(color=mcolor, icon=micon, prefix="fa"),
                popup=folium.Popup(
                    f"<b style='font-size:12px'>{r.get('address_en','')} </b><br>"
                    f"<span style='color:#666'>{legend_val}</span><br>"
                    f"Walk radius: <b>{radius_m} m</b><br>"
                    f"Materials: {cat_dots}",
                    max_width=240),
                tooltip=f"{legend_val}",
            ).add_to(cluster)
        cluster.add_to(rcp_grp)
        rcp_grp.add_to(m)

    # ── Layer 4: GREEN@ hub markers ───────────────────────────────────
    if show_green:
        green_grp = folium.FeatureGroup(name="GREEN@ Hubs (12)", show=True)
        for _, r in green_df.iterrows():
            folium.Marker(
                [r["lat"], r["lon"]],
                icon=folium.Icon(color="darkgreen", icon="leaf", prefix="fa"),
                popup=folium.Popup(
                    f"<b>{r['name']}</b><br>{r['name_zh']}<br>"
                    f"Full-service · Walk catchment: 2 000 m",
                    max_width=210),
                tooltip=f"GREEN@ {r['name']}",
            ).add_to(green_grp)
        green_grp.add_to(m)

    # ── Layer 5: Underserved estates ─────────────────────────────────
    if show_underserved:
        density_threshold = estates_df["no_of_flats"].median()
        _eff_dists = composite_dists if composite_dists is not None else green_dists
        us_mask = ((_eff_dists > service_radius_m) &
                   (estates_df["no_of_flats"].values > density_threshold))
        underserved = estates_df[us_mask].copy()
        underserved["walk_green_km"]    = green_dists[us_mask] / 1000
        underserved["composite_eff_km"] = _eff_dists[us_mask] / 1000

        us_grp = folium.FeatureGroup(name="Underserved Estates", show=True)
        for _, r in underserved.iterrows():
            # Severity: darker = further walk distance
            sev   = min(r["composite_eff_km"] / 5.0, 1.0)
            alpha = 0.45 + 0.35 * sev
            folium.Circle(
                [r["lat"], r["lng"]], radius=220,
                color="#f85149", fill=True, fill_color="#f85149",
                fill_opacity=alpha, weight=2,
                tooltip=(f"⚠ {r['estate_name']} — {r['district_name']}<br>"
                         f"{int(r['no_of_flats'])} flats<br>"
                         f"Walk to GREEN@: {r['walk_green_km']:.1f} km<br>"
                         f"Composite walk: {r['composite_eff_km']:.1f} km"),
            ).add_to(us_grp)
        us_grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────
    # 5 category rows + station tier rows — compact and scrollable
    cat_rows = "".join(
        f'<div class="legend-row"><div class="legend-dot" style="background:{col}"></div>'
        f'<span>{cat}</span></div>\n'
        for cat, col in CATEGORY_COLOR.items()
    )
    tier_rows = "".join(
        f'<div class="legend-row" style="font-size:10px;color:#5a8f5a">'
        f'<div class="legend-sq" style="border:1px solid #3fb950;border-radius:50%;background:transparent"></div>'
        f'<span>{lbl} — {r_m} m</span></div>\n'
        for lbl, r_m in [("GREEN@ Hub", 2000), ("Recycling Station", 800),
                          ("NGO Point", 600), ("Smart Bin", 400),
                          ("Public Bin", 200)]
    )
    walk_note = (
        "★ GREEN@ zones = real OSMnx walk isochrones<br>"
        if walk_adj else
        "★ Zones = radius circles (OSMnx walk graph<br>&nbsp;&nbsp;not loaded — isochrones unavailable)"
    )

    legend_html = f"""
    <style>
      .leaflet-control-layers {{
        overflow:visible!important;max-height:none!important;overflow-y:visible!important;
        background:rgba(10,15,10,0.85)!important;backdrop-filter:blur(10px)!important;
        border:1px solid rgba(62,185,80,0.25)!important;border-radius:12px!important;
        box-shadow:0 4px 24px rgba(0,0,0,0.5)!important;
        color:#90c890!important;font-family:'DM Sans',sans-serif!important;font-size:12px!important;
      }}
      .leaflet-control-layers-list{{overflow:visible!important;max-height:none!important}}
      .leaflet-control-layers-scrollbar{{overflow:visible!important}}
      .leaflet-control-layers-base label,.leaflet-control-layers-overlays label{{color:#90c890!important;font-size:12px!important}}
      .leaflet-control-layers-separator{{border-top:1px solid rgba(62,185,80,0.2)!important}}
      .map-legend{{
        position:absolute;bottom:36px;left:14px;z-index:1000;
        background:rgba(10,15,10,0.88);backdrop-filter:blur(12px);
        border:1px solid rgba(62,185,80,0.22);border-radius:14px;
        box-shadow:0 8px 32px rgba(0,0,0,0.55);
        padding:14px 16px;min-width:188px;max-height:80vh;overflow-y:auto;
        font-family:'DM Sans',sans-serif;
      }}
      .legend-title{{color:#3fb950;font-size:10px;font-weight:700;
        letter-spacing:.13em;text-transform:uppercase;
        margin-bottom:9px;padding-bottom:6px;border-bottom:1px solid rgba(62,185,80,0.2)}}
      .legend-row{{display:flex;align-items:center;gap:9px;margin:5px 0;color:#8ab88a;font-size:11px}}
      .legend-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
      .legend-sq{{width:10px;height:10px;border-radius:3px;flex-shrink:0}}
      .legend-hr{{border:none;border-top:1px solid rgba(62,185,80,0.12);margin:7px 0}}
      .legend-section{{color:#5a8f5a;font-size:9px;font-weight:700;
        text-transform:uppercase;letter-spacing:.1em;margin:8px 0 3px}}
      .legend-note{{color:#4a6e4a;font-size:9.5px;margin-top:8px;line-height:1.6}}
      .legend-toggle{{float:right;background:none;border:none;color:#3fb950;font-size:14px;cursor:pointer}}
    </style>
    <script>
      (function(){{
        function patch(){{
          document.querySelectorAll('[id^="macro_element_div"]').forEach(function(el){{
            el.id="population_distribution_map";
            el.setAttribute("aria-label","Population distribution and recycling accessibility map");
          }});
          var labels=document.querySelectorAll(".leaflet-control-layers-base label span");
          labels.forEach(function(s){{
            var t=s.textContent.trim().toLowerCase().replace(/[ \\t]+/g,"");
            if(t==="cartodbdarkmatter"||t==="cartodbdark matter")s.textContent="Base map";
          }});
        }}
        document.readyState==="loading"?document.addEventListener("DOMContentLoaded",patch):patch();
        setTimeout(patch,300);
      }})();
    </script>
    <div class="map-legend" id="rl_panel">
      <div class="legend-title">Legend
        <button class="legend-toggle" onclick="var b=document.getElementById('rlb');
          b.style.display=b.style.display==='none'?'block':'none';
          this.textContent=b.style.display==='none'?'+':'\u2212';">&#8722;</button>
      </div>
      <div id="rlb">
        <div class="legend-row">
          <div class="legend-sq" style="background:rgba(255,238,0,0.65)"></div>
          <span>High residential density</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:#f85149"></div>
          <span>Underserved estate (walk)</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:transparent;border:2px solid #3fb950;width:9px;height:9px"></div>
          <span>GREEN@ Hub isochrone</span>
        </div>
        <hr class="legend-hr">
        <div class="legend-section">♻ Material categories</div>
        {cat_rows}
        <hr class="legend-hr">
        <div class="legend-section">🚶 Walk radius by tier</div>
        {tier_rows}
        <div class="legend-note">{walk_note}</div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m



# ROAD NETWORK — OSMnx (OpenStreetMap) with GeoJSON fallback
# ══════════════════════════════════════════════════════════════════════
#
# PRIMARY path  : OSMnx downloads the HK drive network from OpenStreetMap,
#                 caches it to disk as "hk_road_graph.graphml" so subsequent
#                 runs are instant.  Requires internet on first run.
#
# FALLBACK path : if osmnx is not installed OR the download fails, the app
#                 automatically falls back to the local speed_limit.geojson
#                 (original behaviour, with dashed straight-line segments
#                 for disconnected components).
#
# To force the fallback (e.g. offline demo):
#   set USE_OSMNX = False below.

USE_OSMNX   = True          # ← set False to force GeoJSON fallback
GRAPH_CACHE = dp("hk_road_graph.graphml")   # cached OSMnx graph on disk


@st.cache_resource
def build_road_network_osmnx():
    """
    Download (or load from cache) the Hong Kong driveable road network
    via OSMnx and return a lightweight routing structure.

    Returns
    -------
    adj        : dict  node_osmid → [(neighbour_osmid, weight_m)]
    coords     : dict  node_osmid → (lat, lon)
    grid_index : dict  cell → [node_osmid, ...]
    GRID       : float
    """
    import osmnx as ox
    import networkx as nx

    GRID = 0.01

    # Load from cache or download fresh
    if os.path.exists(GRAPH_CACHE):
        G = ox.load_graphml(GRAPH_CACHE)
    else:
        with st.spinner("Downloading HK road network from OpenStreetMap (one-time, ~15s)…"):
            G = ox.graph_from_place(
                "Hong Kong",
                network_type="drive",
                simplify=True,
            )
            ox.save_graphml(G, GRAPH_CACHE)

    # Build lightweight adj dict + coord lookup (avoids keeping full NetworkX
    # graph in memory across Streamlit reruns)
    adj    = defaultdict(list)
    coords = {}   # osmid → (lat, lon)

    for u, v, data in G.edges(data=True):
        w = data.get("length", 1.0)   # OSMnx stores length in metres
        adj[u].append((v, w))
        adj[v].append((u, w))          # treat as undirected for truck routing

    for node, data in G.nodes(data=True):
        coords[node] = (data["y"], data["x"])   # (lat, lon)

    # Spatial grid for fast nearest-node lookup
    grid_index = defaultdict(list)
    for osmid, (lat, lon) in coords.items():
        cell = (round(lon // GRID * GRID, 3),
                round(lat // GRID * GRID, 3))
        grid_index[cell].append(osmid)

    return dict(adj), coords, grid_index, GRID


@st.cache_resource
def build_road_network_geojson(geojson_path: str):
    """
    Fallback: build road graph from local SPEED_LIMIT GeoJSON.
    Identical to the original implementation.
    """
    ROUND = 5
    GRID  = 0.01
    adj   = defaultdict(list)

    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    for ft in gj["features"]:
        if not ft.get("geometry"):
            continue
        raw = ft["geometry"]["coordinates"]
        pts = [(round(c[0], ROUND), round(c[1], ROUND)) for c in raw]
        for i in range(len(pts) - 1):
            u, v = pts[i], pts[i + 1]
            if u == v:
                continue
            w = haversine_m(u[1], u[0], v[1], v[0])
            adj[u].append((v, w))
            adj[v].append((u, w))

    # Bridge disconnected components (same as before)
    comp_sets = {}; cid = 0; remaining = set(adj.keys())
    while remaining:
        seed = next(iter(remaining)); comp = {seed}; q = [seed]
        while q:
            n = q.pop()
            for nb, _ in adj[n]:
                if nb not in comp: comp.add(nb); q.append(nb)
        comp_sets[cid] = comp; remaining -= comp; cid += 1

    grid_index = defaultdict(list)
    for node in adj:
        cell = (round(node[0] // GRID * GRID, 3),
                round(node[1] // GRID * GRID, 3))
        grid_index[cell].append(node)

    main_cid  = max(comp_sets, key=lambda k: len(comp_sets[k]))
    main_grid = defaultdict(list)
    for n in comp_sets[main_cid]:
        cell = (round(n[0] // GRID * GRID, 3), round(n[1] // GRID * GRID, 3))
        main_grid[cell].append(n)

    def _nearest_in_main(lat, lon):
        cx = round(lon // GRID * GRID, 3); cy = round(lat // GRID * GRID, 3)
        best_d, best_n = float("inf"), None
        for r in range(1, 8):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if abs(dx) < r and abs(dy) < r: continue
                    for n in main_grid.get((round(cx+dx*GRID,3), round(cy+dy*GRID,3)), []):
                        d = haversine_m(lat, lon, n[1], n[0])
                        if d < best_d: best_d, best_n = d, n
            if best_n and best_d < 2000: break
        return best_n, best_d

    for cid_k, nodes in comp_sets.items():
        if cid_k == main_cid: continue
        lons = [n[0] for n in nodes]; lats = [n[1] for n in nodes]
        mn, md = _nearest_in_main(sum(lats)/len(lats), sum(lons)/len(lons))
        if mn and md < 3000:
            cn = min(nodes, key=lambda n: haversine_m(n[1], n[0], mn[1], mn[0]))
            bw = haversine_m(cn[1], cn[0], mn[1], mn[0])
            adj[cn].append((mn, bw)); adj[mn].append((cn, bw))

    # For GeoJSON graph, coords ARE the node keys (lon, lat tuples)
    coords = {n: (n[1], n[0]) for n in adj}
    return dict(adj), coords, grid_index, GRID


def snap_to_road(lat, lon, grid_index, GRID, search_radius=4):
    """Return nearest graph node to (lat, lon) using the grid index."""
    cx = round(lon // GRID * GRID, 3)
    cy = round(lat // GRID * GRID, 3)
    best_d, best_n = float("inf"), None
    for r in range(1, search_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cell = (round(cx + dx * GRID, 3), round(cy + dy * GRID, 3))
                for n in grid_index.get(cell, []):
                    # coords lookup works for both OSMnx (int key) and
                    # GeoJSON (tuple key — grid_index stores the node itself)
                    if isinstance(n, tuple):
                        nd = haversine_m(lat, lon, n[1], n[0])
                    else:
                        nd = haversine_m(lat, lon, n[1], n[0]) if False else                              grid_index.get("__coords__", {}).get(n, (lat,lon))
                        nd = haversine_m(lat, lon, nd[0], nd[1])
                    if nd < best_d:
                        best_d, best_n = nd, n
        if best_n is not None and best_d < 1000:
            break
    return best_n, best_d


def snap_to_road_v2(lat, lon, grid_index, coords, GRID, search_radius=5):
    """
    Unified snap function that works for both OSMnx (int node IDs) and
    GeoJSON (tuple node IDs).  Uses the coords dict for distance calc.
    """
    cx = round(lon // GRID * GRID, 3)
    cy = round(lat // GRID * GRID, 3)
    best_d, best_n = float("inf"), None
    for r in range(1, search_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cell = (round(cx + dx * GRID, 3), round(cy + dy * GRID, 3))
                for n in grid_index.get(cell, []):
                    nlat, nlon = coords[n]
                    d = haversine_m(lat, lon, nlat, nlon)
                    if d < best_d:
                        best_d, best_n = d, n
        if best_n is not None and best_d < 1000:
            break
    return best_n, best_d


def dijkstra_road(adj, coords, source, target):
    """
    Dijkstra on the road graph.  Works for both OSMnx and GeoJSON graphs.

    Returns
    -------
    dist_m : float
    path   : list of [lat, lon]  ready for Folium PolyLine
    """
    dist = {source: 0.0}
    prev = {}
    pq   = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            path = []
            while u in prev:
                path.append(coords[u])
                u = prev[u]
            path.append(coords[source])
            return d, [[lat, lon] for lat, lon in reversed(path)]
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return float("inf"), []


def road_route_or_fallback(adj, coords, grid_index, GRID, lat1, lon1, lat2, lon2):
    """
    Route from (lat1,lon1) to (lat2,lon2) along the road graph.
    Returns (coord_list, dist_m, is_road_route).
    is_road_route=False means a straight-line fallback was used.
    """
    s = snap_to_road_v2(lat1, lon1, grid_index, coords, GRID)
    t = snap_to_road_v2(lat2, lon2, grid_index, coords, GRID)

    if s[0] and t[0]:
        dist_m, path = dijkstra_road(adj, coords, s[0], t[0])
        if path:
            return [[lat1, lon1]] + path + [[lat2, lon2]], dist_m, True

    # Fallback: straight line
    return [[lat1, lon1], [lat2, lon2]], haversine_m(lat1, lon1, lat2, lon2), False


def build_route_map(estates_df, rcp_df, green_df,
                    service_radius_m, n_trucks, green_dists,
                    road_adj, road_coords, road_grid, road_grid_size,
                    composite_dists=None):
    """
    Mobile truck routing for underserved estates (far from GREEN@ hubs).

    Routing algorithm:
      1. Nearest-neighbour heuristic orders stops for each truck.
      2. Each leg is routed via Dijkstra on the road graph (OSMnx or
         GeoJSON fallback).
      3. Solid line = real road path.  Dashed line = straight-line
         fallback used only when no graph path exists (only happens
         with the GeoJSON fallback on disconnected island roads).
         With OSMnx these dashed lines should not appear.
    """
    m = folium.Map(location=HK_CENTER, zoom_start=11,
                   tiles="CartoDB dark_matter", control_scale=True)

    # Background: sparse collection point dots
    for _, r in rcp_df.iloc[::20].iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=2,
                             color="#388bfd", fill=True, fill_opacity=0.3).add_to(m)

    # GREEN@ depots (larger markers)
    for _, r in green_df.iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=9,
                             color="#3fb950", fill=True, fill_opacity=0.85,
                             tooltip=r["name"]).add_to(m)

    density_threshold = estates_df["no_of_flats"].median()
    _eff_dists = composite_dists if composite_dists is not None else green_dists
    us_mask = ((_eff_dists > service_radius_m) &
               (estates_df["no_of_flats"].values > density_threshold))
    underserved = estates_df[us_mask].copy()
    underserved["dist_green_km"] = green_dists[us_mask] / 1000

    if underserved.empty:
        return m, underserved

    pts = underserved[["lat", "lng"]].values.tolist()

    # ── Nearest-neighbour stop ordering ───────────────────────────
    def nn_route(pts):
        unvisited = list(range(len(pts)))
        route = [unvisited.pop(0)]
        while unvisited:
            last = pts[route[-1]]
            d = [haversine_m(last[0], last[1], pts[i][0], pts[i][1])
                 for i in unvisited]
            nxt = unvisited[int(np.argmin(d))]
            route.append(nxt)
            unvisited.remove(nxt)
        return route

    colors = ["#f0883e", "#79c0ff", "#d2a8ff", "#56d364"]
    names  = ["Truck A", "Truck B", "Truck C", "Truck D"]
    assignments = [[] for _ in range(n_trucks)]
    for i, pt in enumerate(pts):
        assignments[i % n_trucks].append(pt)

    green_coords = green_df[["lat", "lon"]].values.tolist()

    for t in range(n_trucks):
        apt = assignments[t]
        if not apt:
            continue
        color, tname = colors[t], names[t]
        ordered = [apt[i] for i in nn_route(apt)]

        # Depot = nearest GREEN@ hub to this truck's cluster centroid
        c_lat = np.mean([p[0] for p in apt])
        c_lng = np.mean([p[1] for p in apt])
        dep_idx = int(np.argmin([haversine_m(c_lat, c_lng, g[0], g[1])
                                  for g in green_coords]))
        depot = green_coords[dep_idx]   # [lat, lon]

        # ── Build full road-following route ───────────────────────
        # waypoints: depot → stop0 → stop1 → … → stopN → depot
        waypoints = [depot] + ordered + [depot]
        total_road_km = 0.0

        for seg_i in range(len(waypoints) - 1):
            a, b = waypoints[seg_i], waypoints[seg_i + 1]
            seg_coords, seg_dist, is_road = road_route_or_fallback(
                road_adj, road_coords, road_grid, road_grid_size,
                a[0], a[1], b[0], b[1]
            )
            total_road_km += seg_dist / 1000

            # Solid line = road-following path (OSMnx or GeoJSON graph)
            # Dashed line = straight-line fallback (disconnected graph only;
            #               should not appear when using OSMnx)
            folium.PolyLine(
                seg_coords,
                color=color,
                weight=4.5 if is_road else 3.0,
                opacity=0.90 if is_road else 0.55,
                dash_array=None if is_road else "10 6",
                tooltip=(f"{tname} — leg {seg_i+1}/{len(waypoints)-1}  "
                         f"({'road' if is_road else 'straight fallback'})  "
                         f"{seg_dist/1000:.1f} km"),
            ).add_to(m)

        # Depot marker
        folium.Marker(
            depot,
            icon=folium.Icon(color="darkgreen", icon="truck", prefix="fa"),
            tooltip=(f"{tname} depot: {green_df.iloc[dep_idx]['name']}  |  "
                     f"total route {total_road_km:.1f} km"),
        ).add_to(m)

        # Stop markers
        for stop_i, stop in enumerate(ordered, 1):
            folium.CircleMarker(
                stop, radius=8, color=color,
                fill=True, fill_color=color, fill_opacity=0.9,
                tooltip=f"{tname} — Stop {stop_i}",
            ).add_to(m)

    return m, underserved


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
with st.spinner("Loading HK EPD datasets…"):
    rcp_df     = load_recycling_points()
    green_df   = load_green_stations()
    estates_df = load_housing_estates()
    stats_df   = load_waste_stats()
    cost_data  = load_cost_data()

# Build road network graph (cached — only runs once per session)
# ── Road network: try OSMnx first, fall back to local GeoJSON ────────
_osmnx_ok = False
if USE_OSMNX:
    try:
        import osmnx as ox          # noqa: F401
        with st.spinner("Loading road network (OSMnx)…"):
            road_adj, road_coords, road_grid, road_grid_size = build_road_network_osmnx()
        _osmnx_ok = True
    except Exception as _e:
        st.warning(f"OSMnx unavailable ({_e}) — falling back to local GeoJSON road data.")

if not _osmnx_ok:
    _ROAD_GEOJSON = dp("speed_limit.geojson")
    if not os.path.exists(_ROAD_GEOJSON):
        _ROAD_GEOJSON = dp("Transportation_TNM_20260311_gdb_SPEED_LIMIT_converted.geojson")
    with st.spinner("Building road network from local GeoJSON…"):
        road_adj, road_coords, road_grid, road_grid_size = build_road_network_geojson(_ROAD_GEOJSON)

# Pre-compute distances (hashed/cached by tuple of coords)
green_dists = compute_distances_to_green(
    tuple(estates_df["lat"].tolist()),
    tuple(estates_df["lng"].tolist()),
    tuple(green_df["lat"].tolist()),
    tuple(green_df["lon"].tolist()),
)

# ── Pedestrian walk network (OSMnx) ──────────────────────────────────
# Used for (1) walk isochrone polygons in build_main_map and
#          (2) true walk distances in composite underserved detection.
# Loaded lazily — falls back to haversine×1.3 if osmnx unavailable.
walk_adj = walk_coords = walk_grid = walk_GRID = None
if USE_OSMNX:
    try:
        with st.spinner("Loading pedestrian walk network (OSMnx)…"):
            walk_adj, walk_coords, walk_grid, walk_GRID = build_walk_network_osmnx()
        if walk_adj:
            st.sidebar.success("✅ Walk network loaded — isochrones active", icon="🚶")
    except Exception as _we:
        st.sidebar.info(f"Walk isochrones unavailable ({_we}) — using radius fallback.")

# Small/medium premium stations
_small_premium = rcp_df[rcp_df["legend"].isin(
    ["Recycling Stations/Recycling Stores", "NGO Collection Points"]
)]

# Use true walk distances if walk graph is available; otherwise haversine
if walk_adj:
    with st.spinner("Computing pedestrian walk distances to GREEN@ hubs…"):
        green_dists = compute_walk_distances_to_green(
            tuple(estates_df["lat"].tolist()),
            tuple(estates_df["lng"].tolist()),
            tuple(green_df["lat"].tolist()),
            tuple(green_df["lon"].tolist()),
            walk_adj, walk_coords, walk_grid, walk_GRID,
        )
        small_station_dists = compute_walk_distances_to_small(
            tuple(estates_df["lat"].tolist()),
            tuple(estates_df["lng"].tolist()),
            tuple(_small_premium["lat"].tolist()),
            tuple(_small_premium["lon"].tolist()),
            walk_adj, walk_coords, walk_grid, walk_GRID,
        )
else:
    small_station_dists = compute_distances_to_small_stations(
        tuple(estates_df["lat"].tolist()),
        tuple(estates_df["lng"].tolist()),
        tuple(_small_premium["lat"].tolist()),
        tuple(_small_premium["lon"].tolist()),
    )

# Composite effective walk distance used everywhere for underserved detection
composite_dists = compute_composite_accessibility(green_dists, small_station_dists)

density_threshold = estates_df["no_of_flats"].median()

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:2.2rem'>♻️</div>
        <div style='color:#3fb950;font-weight:700;font-size:1.05rem;letter-spacing:.04em'>
            HK Recycling Access
        </div>
        <div style='color:#5a8f5a;font-size:0.72rem;margin-top:3px'>
            Hackathon · Real EPD Data
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ⚙ Layers")
    show_heat        = st.checkbox("🌡 Population density (flats)",   value=True)
    show_rcp         = st.checkbox("📍 All collection points (8,796)", value=True)
    show_green       = st.checkbox("🌿 GREEN@ Hubs (12)",              value=True)
    show_coverage    = st.checkbox("⭕ Service radius overlay",         value=True)
    show_underserved = st.checkbox("🔴 Underserved estates",            value=True)

    st.markdown("## 🎛 Parameters")
    # Service radius for GREEN@ hubs — meaningful range given hub coverage areas
    service_radius = st.slider(
        "GREEN@ hub service radius (m)", 500, 5000, 2000, 250,
        help="Estates beyond this distance from any GREEN@ hub are flagged as underserved.")
    rcp_filter = st.selectbox("Point type filter", ["All", "Basic", "Premium"])

    # Live count — uses composite effective distance (GREEN@ + small stations)
    _us = ((composite_dists > service_radius) &
           (estates_df["no_of_flats"].values > density_threshold))
    st.markdown(f"""
    <div class='fact-card' style='margin-top:12px'>
        At <b>{service_radius/1000:.1f}km</b> composite threshold:<br>
        <b style='color:#d4f0d4;font-size:1.3rem'>{int(_us.sum())}</b> underserved estates<br>
        <span style='color:#5a8f5a'>{round(100*_us.sum()/len(estates_df),1)}%
        of high-density estates</span><br>
        <span style='color:#3a6a3a;font-size:0.72rem'>
        Model: GREEN@ hub (weight 1.0) +<br>small stations (weight 0.5)
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Legend")
    st.markdown("""
    <div style='font-size:0.81rem;line-height:2.2;color:#5a8f5a'>
        <span style='color:#388bfd'>●</span> Basic bin<br>
        <span style='color:#2ea043'>●</span> Smart bin / Station<br>
        <span style='color:#3fb950'>★</span> GREEN@ Hub<br>
        <span style='color:#f85149'>●</span> Underserved estate<br>
        <span style='color:#ffee00'>▓</span> High housing density
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.section-divider {
    border: none;
    border-top: 1px solid #1e3a1e;
    margin: 40px 0 28px 0;
}
.section-header {
    color: #3fb950 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    margin-bottom: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# Quick-nav anchor links
st.markdown("""
<div style='background:#111e11;border:1px solid #1e3a1e;border-radius:10px;
     padding:10px 18px;margin-bottom:24px;display:flex;gap:18px;flex-wrap:wrap;
     font-size:0.82rem;font-family:"DM Mono",monospace'>
    <a href="#recycling-rate-trend"   style='color:#5a8f5a;text-decoration:none'>📈 Recycling Trend</a>
    <a href="#accessibility-map"      style='color:#5a8f5a;text-decoration:none'>🗺️ Accessibility Map</a>
    <a href="#station-optimisation"   style='color:#5a8f5a;text-decoration:none'>📍 Station Optimisation</a>
    <a href="#mobile-route-simulation" style='color:#5a8f5a;text-decoration:none'>🚛 Route Simulation</a>
    <a href="#strategy-impact"        style='color:#5a8f5a;text-decoration:none'>📊 Strategy Impact</a>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SECTION 1 — RECYCLING TREND
# ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<a name="recycling-rate-trend"></a>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">📈 Recycling Rate Trend</h2>', unsafe_allow_html=True)
if True:
    import plotly.graph_objects as go

    # ── Main title + subtitle ─────────────────────────────────────────
    st.markdown("""
    <div style='margin-bottom: 6px;'>
        <div style='font-size:2rem;font-weight:700;color:#d4f0d4;letter-spacing:-0.02em;
                    font-family:"DM Sans",sans-serif;line-height:1.2'>
            Falling for 15 Years
        </div>
        <div style='font-size:1rem;color:#8ab88a;margin-top:6px;max-width:640px;line-height:1.6'>
            Hong Kong's recycling rate has declined from
            <b style='color:#d4f0d4'>49.3%</b> in 2009 to just
            <b style='color:#f85149'>34.4%</b> in 2024 — a steady erosion
            driven by falling recovery volumes, not rising waste generation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── 3 KPI cards ───────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("2009 Recycling Rate", "49.3%", delta="Peak year baseline")
    with k2:
        st.metric("2024 Recycling Rate", "34.4%", delta="↓ Latest figure", delta_color="inverse")
    with k3:
        st.metric("Overall Change", "−30%", delta="Relative decline since 2009", delta_color="inverse")

    st.markdown("")

    # ── Filled area line chart via Plotly ─────────────────────────────
    chart_df = stats_df.sort_values("year").copy()

    fig_trend = go.Figure()

    # Filled area under the line
    fig_trend.add_trace(go.Scatter(
        x=chart_df["year"],
        y=chart_df["recycling_rate"],
        mode="lines+markers",
        line=dict(color="#3fb950", width=2.5, shape="spline"),
        marker=dict(size=6, color="#3fb950", line=dict(width=1.5, color="#0a0f0a")),
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.12)",
        name="Recycling rate",
        hovertemplate="<b>%{x}</b><br>Rate: %{y:.1f}%<extra></extra>",
    ))

    # Annotation: 2009
    fig_trend.add_annotation(
        x=2009, y=chart_df.loc[chart_df["year"]==2009, "recycling_rate"].values[0],
        text="<b>49.3%</b><br>2009",
        showarrow=True, arrowhead=2, arrowcolor="#5a8f5a", arrowwidth=1.5,
        ax=28, ay=-40,
        font=dict(size=11, color="#d4f0d4", family="DM Mono, monospace"),
        bgcolor="rgba(17,30,17,0.85)", bordercolor="#3fb950", borderwidth=1, borderpad=5,
    )
    # Annotation: trough ~2020
    trough_yr = int(chart_df.loc[chart_df["recycling_rate"].idxmin(), "year"])
    trough_val = chart_df["recycling_rate"].min()
    fig_trend.add_annotation(
        x=trough_yr, y=trough_val,
        text=f"<b>{trough_val:.1f}%</b><br>{trough_yr} trough",
        showarrow=True, arrowhead=2, arrowcolor="#f85149", arrowwidth=1.5,
        ax=0, ay=-48,
        font=dict(size=11, color="#f85149", family="DM Mono, monospace"),
        bgcolor="rgba(26,17,0,0.85)", bordercolor="#f85149", borderwidth=1, borderpad=5,
    )
    # Annotation: 2024
    fig_trend.add_annotation(
        x=2024, y=chart_df.loc[chart_df["year"]==2024, "recycling_rate"].values[0],
        text="<b>34.4%</b><br>2024",
        showarrow=True, arrowhead=2, arrowcolor="#5a8f5a", arrowwidth=1.5,
        ax=-32, ay=-40,
        font=dict(size=11, color="#d4f0d4", family="DM Mono, monospace"),
        bgcolor="rgba(17,30,17,0.85)", bordercolor="#3fb950", borderwidth=1, borderpad=5,
    )

    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,30,17,0.45)",
        height=320,
        margin=dict(l=8, r=8, t=24, b=8),
        showlegend=False,
        xaxis=dict(
            tickfont=dict(size=11, color="#5a8f5a", family="DM Mono, monospace"),
            showgrid=False, zeroline=False,
            tickmode="linear", dtick=2,
        ),
        yaxis=dict(
            ticksuffix="%",
            tickfont=dict(size=11, color="#5a8f5a", family="DM Mono, monospace"),
            showgrid=True,
            gridcolor="rgba(30,58,30,0.5)",
            zeroline=False,
            range=[0, 58],
        ),
        hoverlabel=dict(bgcolor="#111e11", bordercolor="#3fb950",
                        font=dict(color="#d4f0d4", size=11)),
    )

    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    # ── Bottom insight box (light red highlight) ──────────────────────
    st.markdown("""
    <div style='background:rgba(248,81,73,0.08);border:1px solid rgba(248,81,73,0.35);
         border-left:4px solid #f85149;border-radius:8px;
         padding:14px 18px;margin-top:6px;'>
        <span style='color:#f85149;font-weight:700;font-size:0.95rem'>⚠ The Core Contradiction</span><br>
        <span style='color:#e8b4b0;font-size:0.9rem;line-height:1.7'>
            Hong Kong generates <b>less</b> waste today than in 2009 — yet it recycles a
            <b>far smaller proportion</b> of it. The culprit is not behaviour, but infrastructure:
            only <b>12 GREEN@ premium hubs</b> exist city-wide, leaving millions of flat-dwellers
            beyond convenient reach of full-service recycling. Closing this accessibility gap
            is the lever most likely to reverse the 15-year decline.
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Full statistics table (2009–2024)"):
        show = stats_df[["year","generation_q","recovery_q","disposal_q","recycling_rate"]].copy()
        show.columns = ["Year", "Generated (kg)", "Recovered (kg)", "Disposed (kg)", "Rate (%)"]
        show["Rate (%)"] = show["Rate (%)"].round(1)
        st.dataframe(show, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# SECTION 2 — ACCESSIBILITY MAP
# ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<a name="accessibility-map"></a>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">🗺️ Accessibility Map</h2>', unsafe_allow_html=True)
if True:
    # Layout: map (wider) | chart (narrower)
    col_map, col_info = st.columns([4, 1])

    with col_map:
        with st.spinner("Rendering map…"):
            main_map = build_main_map(
                rcp_df, green_df, estates_df,
                service_radius_m=service_radius,
                show_heat=show_heat, show_rcp=show_rcp, show_green=show_green,
                show_coverage=show_coverage, show_underserved=show_underserved,
                rcp_filter=rcp_filter,
                green_dists=green_dists,
                composite_dists=composite_dists,
                walk_adj=walk_adj, walk_coords=walk_coords,
                walk_grid=walk_grid, walk_GRID=walk_GRID,
            )
            st_folium(main_map, width="100%", height=640, returned_objects=[])

        st.markdown("")
        st.markdown("""
        <div style='background:#1a1100;border:1px solid #3d2b00;border-left:4px solid #d29922;
             border-radius:8px;padding:14px 18px;margin:8px 0;'>
            <span style='color:#d29922;font-weight:700;font-size:1.05rem'>⚡ Key Insight</span><br>
            <span style='color:#c8a840;font-size:0.97rem;line-height:1.7'>
                Despite <b>8,796 basic collection bins</b> — dense enough that every housing estate
                sits within 500m of at least one — only <b>12 GREEN@ flagship hubs</b> offer
                full-service recycling with e-waste, food waste, clothes and professional sorting.
                Residents beyond 2km of a GREEN@ hub have significantly lower convenient access
                to comprehensive recycling services.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Right column: Underserved Rate by District chart ───────────────
    with col_info:
        import plotly.graph_objects as go

        # Compute underserved rate per district
        dist_total = rcp_df.groupby("district_id").size().reset_index(name="total")

        # Merge estate underserved data with districts — use composite effective distance
        density_med_val = estates_df["no_of_flats"].median()
        us_mask_map = (composite_dists > service_radius) & (estates_df["no_of_flats"].values > density_med_val)
        us_estates_map = estates_df[us_mask_map].copy()

        # Count underserved estates per district
        if "district_name" in estates_df.columns:
            total_by_dist = estates_df.groupby("district_name").size().reset_index(name="total_estates")
            us_by_dist    = us_estates_map.groupby("district_name").size().reset_index(name="us_estates")
            dist_us = total_by_dist.merge(us_by_dist, on="district_name", how="left").fillna(0)
            dist_us["us_rate"] = (dist_us["us_estates"] / dist_us["total_estates"].replace(0, 1) * 100).round(1)
            dist_us = dist_us.sort_values("us_rate", ascending=True)
            y_col, x_col = "district_name", "us_rate"
            x_label = "Underserved rate (%)"
        else:
            # Fallback: use collection points count as proxy
            dist_us = dist_total.rename(columns={"district_id": "district", "total": "us_rate"})
            dist_us = dist_us.sort_values("us_rate", ascending=True)
            y_col, x_col = "district", "us_rate"
            x_label = "Collection points"

        # Red gradient bars
        max_val = dist_us[x_col].max() if dist_us[x_col].max() > 0 else 1
        bar_colours_red = [
            f"rgba({int(200 + 48*(v/max_val))}, {int(60 - 40*(v/max_val))}, {int(60 - 40*(v/max_val))}, 0.85)"
            for v in dist_us[x_col]
        ]

        fig_us = go.Figure(go.Bar(
            x=dist_us[x_col],
            y=dist_us[y_col],
            orientation="h",
            marker=dict(color=bar_colours_red, line=dict(width=0)),
            text=[f"{v:.0f}%" if x_col == "us_rate" else f"{int(v):,}"
                  for v in dist_us[x_col]],
            textposition="outside",
            textfont=dict(size=10, color="#f85149", family="DM Mono, monospace"),
            hovertemplate="<b>%{y}</b><br>" + x_label + ": %{x:.1f}<extra></extra>",
        ))

        fig_us.update_layout(
            title=dict(
                text="Underserved Rate by District",
                font=dict(size=12, color="#f85149", family="DM Sans, sans-serif"),
                x=0, xanchor="left",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,10,10,0.5)",
            margin=dict(l=4, r=40, t=36, b=8),
            height=600,
            xaxis=dict(
                title=dict(text=x_label, font=dict(size=9, color="#8a5a5a")),
                showgrid=True,
                gridcolor="rgba(58,20,20,0.7)",
                gridwidth=1,
                zeroline=False,
                tickfont=dict(size=9, color="#8a5a5a"),
                showticklabels=False,
            ),
            yaxis=dict(
                tickfont=dict(size=10, color="#e8a0a0", family="DM Sans, sans-serif"),
                showgrid=False,
                automargin=True,
            ),
            hoverlabel=dict(
                bgcolor="#1a0a0a",
                bordercolor="#f85149",
                font=dict(color="#f0c0c0", size=11),
            ),
        )

        st.plotly_chart(fig_us, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════
# ANALYTICAL FUNCTIONS — Station optimisation & impact model
# ══════════════════════════════════════════════════════════════════════

@st.cache_data
def compute_underserved_clusters(estate_lats, estate_lngs, estate_flats,
                                  estate_names, estate_districts,
                                  green_lats, green_lons,
                                  small_lats=(), small_lons=(),
                                  threshold_m=2000, grid_deg=0.05,
                                  _walk_adj=None, _walk_coords=None,
                                  _walk_grid=None, _walk_GRID=None):
    """
    Cluster underserved high-density estates and return ranked candidate
    station locations.

    Walk-distance model
    -------------------
    When the OSMnx walk graph is available (_walk_adj is not None), distances
    are true pedestrian network distances via Dijkstra.  Otherwise falls back
    to haversine × 1.3 circuity factor.

    Composite effective distance = min(d_green/1.0, d_small/0.5):
      - GREEN@ hub full weight 1.0
      - Small/NGO station half weight 0.5 (partial service)

    threshold_m : composite walk distance above which an estate is underserved
    grid_deg    : ~5 km spatial grid for clustering
    """
    green_coords = list(zip(green_lats, green_lons))
    small_coords = list(zip(small_lats, small_lons))

    rows = []
    for i, (lat, lng) in enumerate(zip(estate_lats, estate_lngs)):
        # Green distance — walk or fallback
        if _walk_adj is not None:
            d_green = walk_dist_to_nearest(lat, lng, green_coords,
                                           _walk_adj, _walk_coords,
                                           _walk_grid, _walk_GRID)
        else:
            d_green = min(haversine_m(lat, lng, g[0], g[1])
                          for g in green_coords) * 1.3
        # Small station distance — walk or fallback
        if small_coords:
            if _walk_adj is not None:
                d_small = walk_dist_to_nearest(lat, lng, small_coords,
                                               _walk_adj, _walk_coords,
                                               _walk_grid, _walk_GRID)
            else:
                d_small = min(haversine_m(lat, lng, s[0], s[1])
                              for s in small_coords) * 1.3
        else:
            d_small = 1e9
        d_eff = min(d_green / 1.0, d_small / 0.5)
        rows.append({
            "lat":         lat,  "lng":        lng,
            "no_of_flats": estate_flats[i],
            "estate_name": estate_names[i],
            "district":    estate_districts[i],
            "dist_green_m": d_green,
            "dist_small_m": d_small,
            "dist_eff_m":   d_eff,
        })
    df = pd.DataFrame(rows)

    density_med = df["no_of_flats"].median()
    us = df[(df["dist_eff_m"] > threshold_m) &
            (df["no_of_flats"] > density_med)].copy()
    us["dist_green_km"] = us["dist_green_m"] / 1000
    us["dist_eff_km"]   = us["dist_eff_m"]   / 1000

    us["grid_lat"] = (us["lat"] // grid_deg * grid_deg).round(3)
    us["grid_lon"] = (us["lng"] // grid_deg * grid_deg).round(3)
    clusters = (us.groupby(["grid_lat", "grid_lon"])
                  .agg(n_estates    =("lat",          "count"),
                       total_flats  =("no_of_flats",  "sum"),
                       centroid_lat =("lat",          "mean"),
                       centroid_lon =("lng",          "mean"),
                       avg_dist_km  =("dist_eff_km",  "mean"),
                       max_dist_km  =("dist_eff_km",  "max"))
                  .reset_index()
                  .sort_values("total_flats", ascending=False)
                  .reset_index(drop=True))
    clusters.index += 1
    clusters["priority_score"] = (
        clusters["total_flats"] / clusters["total_flats"].max() * 0.6 +
        clusters["avg_dist_km"] / clusters["avg_dist_km"].max() * 0.4
    ).round(3)
    return clusters, us


def simulate_station_scenario(all_estate_lats, all_estate_lngs,
                               all_estate_flats, green_coords,
                               new_station_coords, threshold_m=2000,
                               small_coords=None,
                               walk_adj=None, walk_coords=None,
                               walk_grid=None, walk_GRID=None):
    """
    Recompute coverage metrics given existing GREEN@ + new candidate coords.

    Uses true walk distances when the OSMnx walk graph is provided;
    falls back to haversine × 1.3 otherwise.
    """
    combined_green = list(green_coords) + list(new_station_coords)

    if walk_adj is not None:
        green_d = np.array([
            walk_dist_to_nearest(lat, lng, combined_green,
                                 walk_adj, walk_coords, walk_grid, walk_GRID)
            for lat, lng in zip(all_estate_lats, all_estate_lngs)
        ])
        if small_coords and len(small_coords) > 0:
            small_d = np.array([
                walk_dist_to_nearest(lat, lng, small_coords,
                                     walk_adj, walk_coords, walk_grid, walk_GRID)
                for lat, lng in zip(all_estate_lats, all_estate_lngs)
            ])
        else:
            small_d = np.full(len(all_estate_lats), 1e9)
    else:
        green_d = np.array([
            min(haversine_m(lat, lng, g[0], g[1]) for g in combined_green) * 1.3
            for lat, lng in zip(all_estate_lats, all_estate_lngs)
        ])
        if small_coords and len(small_coords) > 0:
            small_d = np.array([
                min(haversine_m(lat, lng, s[0], s[1]) for s in small_coords) * 1.3
                for lat, lng in zip(all_estate_lats, all_estate_lngs)
            ])
        else:
            small_d = np.full(len(all_estate_lats), 1e9)

    dists = np.minimum(green_d / 1.0, small_d / 0.5)
    flats = np.array(all_estate_flats)
    total_flats = flats.sum()
    density_med = np.median(flats)
    us_mask = (dists > threshold_m) & (flats > density_med)

    bands = [(500, 1.0), (1000, 0.8), (2000, 0.6), (3000, 0.3)]
    wscore = sum(flats[dists <= r].sum() * w for r, w in bands) / total_flats

    return {
        "n_underserved":     int(us_mask.sum()),
        "flats_covered_2km": int(flats[dists <= 2000].sum()),
        "pct_covered_2km":   round(100 * flats[dists <= 2000].sum() / total_flats, 1),
        "avg_dist_km":       round(dists.mean() / 1000, 2),
        "weighted_score":    round(wscore, 3),
        "dists":             dists,
    }


def build_station_map(green_df, clusters_df, all_estates_df,
                      n_new_stations, threshold_m, green_dists_arr,
                      composite_dists_arr=None):
    """
    Folium map for Tab 4: existing GREEN@ + candidate new stations
    + underserved clusters + coverage rings.
    """
    m = folium.Map(location=HK_CENTER, zoom_start=11,
                   tiles="CartoDB dark_matter", control_scale=True)

    green_coords = green_df[["lat", "lon"]].values.tolist()
    density_med  = all_estates_df["no_of_flats"].median()
    _eff_dists   = composite_dists_arr if composite_dists_arr is not None else green_dists_arr
    us_mask      = ((_eff_dists > threshold_m) &
                    (all_estates_df["no_of_flats"].values > density_med))
    underserved  = all_estates_df[us_mask]

    # ── Underserved estates (red) ──────────────────────────────────
    for _, r in underserved.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lng"]], radius=6,
            color="#f85149", fill=True, fill_color="#f85149",
            fill_opacity=0.55, weight=1.5,
            tooltip=f"⚠ {r['estate_name']} — {int(r['no_of_flats'])} flats",
        ).add_to(m)

    # ── Existing GREEN@ stations (green stars) ─────────────────────
    for _, r in green_df.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=10,
            color="#3fb950", fill=True, fill_opacity=0.9,
            tooltip=f"✅ {r['name']}",
        ).add_to(m)
        folium.Circle(
            [r["lat"], r["lon"]], radius=threshold_m,
            color="#3fb950", fill=True, fill_opacity=0.04,
            weight=1, dash_array="5 5",
        ).add_to(m)

    # ── Candidate new stations ─────────────────────────────────────
    candidate_colors = ["#f0883e", "#79c0ff", "#d2a8ff", "#56d364", "#ffee00"]
    top = clusters_df.head(n_new_stations)
    for i, (_, row) in enumerate(top.iterrows()):
        color = candidate_colors[i % len(candidate_colors)]
        folium.Marker(
            [row["centroid_lat"], row["centroid_lon"]],
            icon=folium.Icon(color="orange", icon="plus-sign", prefix="glyphicon"),
            tooltip=(f"🆕 Candidate #{i+1} — "
                     f"{int(row['total_flats']):,} flats in catchment  "
                     f"avg {row['avg_dist_km']:.1f}km from existing GREEN@"),
        ).add_to(m)
        folium.Circle(
            [row["centroid_lat"], row["centroid_lon"]],
            radius=threshold_m,
            color=color, fill=True, fill_opacity=0.07,
            weight=2,
        ).add_to(m)

    return m


def compute_cost_model(n_new_stations, n_trucks, cost_data,
                       us_flats, total_flats,
                       baseline_rate=0.344,
                       dist_red_km=0.0,
                       newly_covered_flats=0):
    """
    Unified cost + impact model using recycling_cost.json figures.
    Returns a dict of financial and impact metrics.
    """
    # ── Extract cost parameters ──────────────────────────────────────
    if cost_data:
        sd = cost_data["summary_data"]
        truck_data = cost_data["recycle_truck_cost"][1]   # hybrid truck
        sta_data   = cost_data["recycle_station_cost"][0] # large station
        annual_truck = (sum(truck_data["annual_cost_per_truck"]) / 2)
        annual_sta   = (sum(sta_data["annual_cost_per_station"]) / 2)
        cost_per_ton = sd["public_avg_cost_per_ton"]
        priv_per_ton = sd["private_avg_cost_per_ton"]
    else:
        annual_truck = 990_000
        annual_sta   = 9_240_000
        cost_per_ton = 13_000
        priv_per_ton = 500

    WORKING_DAYS   = 250
    DISPOSAL_SAVED = 450    # HKD per tonne avoided landfill
    RECYCLE_VALUE  = 200    # HKD per tonne recovered material

    annual_mobile_cost  = n_trucks  * annual_truck
    annual_station_cost = n_new_stations * annual_sta
    total_annual_cost   = annual_mobile_cost + annual_station_cost

    # ── Participation model ──────────────────────────────────────────
    rate_lift   = (dist_red_km / 0.5) * 0.03
    access_lift = (newly_covered_flats / max(total_flats, 1)) * 0.10
    proj_rate   = baseline_rate + rate_lift + access_lift

    extra_flats     = int(total_flats * (proj_rate - baseline_rate))
    extra_tonnes_yr = round(extra_flats * 0.3 * 365 * 0.10)
    financial_benefit = extra_tonnes_yr * (DISPOSAL_SAVED + RECYCLE_VALUE)

    return {
        "annual_mobile_cost":   annual_mobile_cost,
        "annual_station_cost":  annual_station_cost,
        "total_annual_cost":    total_annual_cost,
        "proj_rate":            round(proj_rate * 100, 1),
        "extra_flats":          extra_flats,
        "extra_tonnes_yr":      extra_tonnes_yr,
        "financial_benefit":    financial_benefit,
        "net_position":         financial_benefit - total_annual_cost,
        "cost_per_ton":         cost_per_ton,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 3 — STATION OPTIMISATION
# ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<a name="station-optimisation"></a>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">📍 Station Optimisation</h2>', unsafe_allow_html=True)
if True:
    st.markdown("""
    <div style='margin-bottom:14px'>
        <h3 style='margin:0'>📍 Optimal Station Placement</h3>
        <p style='color:#5a8f5a;font-size:0.84rem;margin:4px 0 0'>
            Identify the highest-impact locations for new GREEN@ recycling hubs
            by clustering underserved estates and ranking by population catchment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────
    t4c1, t4c2, t4c3 = st.columns(3)
    with t4c1:
        t4_threshold = st.slider(
            "Underserved threshold (km)", 1.0, 5.0, 2.0, 0.5,
            key="t4_thresh",
            help="Estates beyond this distance from any GREEN@ hub are flagged as underserved.")
    with t4c2:
        t4_n_stations = st.selectbox(
            "New stations to add", [1, 2, 3, 5], index=2, key="t4_nst")
    with t4c3:
        t4_show_rings = st.checkbox("Show coverage rings", value=True, key="t4_rings")

    t4_threshold_m = int(t4_threshold * 1000)

    # ── Compute clusters ──────────────────────────────────────────────
    clusters_df, us_df = compute_underserved_clusters(
        tuple(estates_df["lat"].tolist()),
        tuple(estates_df["lng"].tolist()),
        tuple(estates_df["no_of_flats"].tolist()),
        tuple(estates_df["estate_name"].tolist()),
        tuple(estates_df["district_name"].tolist()),
        tuple(green_df["lat"].tolist()),
        tuple(green_df["lon"].tolist()),
        small_lats=tuple(_small_premium["lat"].tolist()),
        small_lons=tuple(_small_premium["lon"].tolist()),
        threshold_m=t4_threshold_m,
        _walk_adj=walk_adj, _walk_coords=walk_coords,
        _walk_grid=walk_grid, _walk_GRID=walk_GRID,
    )

    # ── Scenario comparison metrics ───────────────────────────────────
    g_coords = list(zip(green_df["lat"], green_df["lon"]))
    _sm_coords = list(zip(_small_premium["lat"], _small_premium["lon"]))
    all_lats = estates_df["lat"].tolist()
    all_lngs = estates_df["lng"].tolist()
    all_flats = estates_df["no_of_flats"].tolist()
    total_flats_all = sum(all_flats)

    base_metrics = simulate_station_scenario(
        all_lats, all_lngs, all_flats, g_coords, [], t4_threshold_m,
        small_coords=_sm_coords,
        walk_adj=walk_adj, walk_coords=walk_coords,
        walk_grid=walk_grid, walk_GRID=walk_GRID)

    top_n = clusters_df.head(t4_n_stations)
    new_coords = list(zip(top_n["centroid_lat"], top_n["centroid_lon"]))
    new_metrics = simulate_station_scenario(
        all_lats, all_lngs, all_flats, g_coords, new_coords, t4_threshold_m,
        small_coords=_sm_coords,
        walk_adj=walk_adj, walk_coords=walk_coords,
        walk_grid=walk_grid, walk_GRID=walk_GRID)

    newly_covered = new_metrics["flats_covered_2km"] - base_metrics["flats_covered_2km"]
    dist_red      = base_metrics["avg_dist_km"] - new_metrics["avg_dist_km"]

    # ── Summary metrics row ───────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Underserved estates",
                  base_metrics["n_underserved"],
                  delta=f"→ {new_metrics['n_underserved']} after +{t4_n_stations}",
                  delta_color="inverse")
    with s2:
        st.metric("Estates within threshold",
                  f"{base_metrics['pct_covered_2km']}%",
                  delta=f"+{new_metrics['pct_covered_2km']-base_metrics['pct_covered_2km']:.1f}pp")
    with s3:
        st.metric("Avg dist to GREEN@",
                  f"{base_metrics['avg_dist_km']} km",
                  delta=f"→ {new_metrics['avg_dist_km']} km")
    with s4:
        st.metric("Weighted coverage score",
                  f"{base_metrics['weighted_score']}",
                  delta=f"+{new_metrics['weighted_score']-base_metrics['weighted_score']:.3f}")
    with s5:
        st.metric("Flats newly covered",
                  f"{newly_covered:,}",
                  delta=f"{round(100*newly_covered/total_flats_all,1)}% of all flats")

    st.markdown("")

    # ── Map + table ───────────────────────────────────────────────────
    map_col, tbl_col = st.columns([3, 1])

    with map_col:
        # Recompute green_dists and composite_dists for current threshold display
        gc = list(zip(green_df["lat"], green_df["lon"]))
        gd = np.array([min(haversine_m(lat, lng, g[0], g[1]) for g in gc)
                       for lat, lng in zip(all_lats, all_lngs)])
        # Use pre-computed composite_dists (already reflects small stations)
        station_map = build_station_map(
            green_df, clusters_df, estates_df,
            n_new_stations=t4_n_stations,
            threshold_m=t4_threshold_m,
            green_dists_arr=gd,
            composite_dists_arr=composite_dists,
        )
        st_folium(station_map, width="100%", height=580, returned_objects=[])

    with tbl_col:
        st.markdown("### Top candidates")
        disp = clusters_df.head(8)[
            ["centroid_lat", "centroid_lon",
             "n_estates", "total_flats",
             "avg_dist_km", "priority_score"]
        ].copy()
        disp.columns = ["Lat", "Lon", "Estates",
                        "Flats", "Avg dist", "Score"]
        disp["Lat"]      = disp["Lat"].round(4)
        disp["Lon"]      = disp["Lon"].round(4)
        disp["Avg dist"] = disp["Avg dist"].round(2)
        disp["Flats"]    = disp["Flats"].astype(int)
        st.dataframe(disp, hide_index=False,
                     use_container_width=True, height=340)

        st.markdown("### Scenario comparison")
        cmp_chart = pd.DataFrame({
            "Baseline": [
                base_metrics["pct_covered_2km"],
                base_metrics["weighted_score"] * 100,
            ],
            f"+{t4_n_stations} stations": [
                new_metrics["pct_covered_2km"],
                new_metrics["weighted_score"] * 100,
            ],
        }, index=["Covered (%)", "Wtd score ×100"])
        st.bar_chart(cmp_chart, color=["#5a8f5a", "#3fb950"], height=180, use_container_width=True)
        dist_cmp = pd.DataFrame({
            "Baseline": [base_metrics["avg_dist_km"]],
            f"+{t4_n_stations} sta.": [new_metrics["avg_dist_km"]],
        }, index=["Avg dist (km)"])
        st.line_chart(dist_cmp, color=["#5a8f5a", "#3fb950"], height=100, use_container_width=True)

    st.markdown("""
    <div class='fact-card'>
        <b>How candidate locations are ranked:</b>
        Priority score = 60% weight on total flats in catchment
        + 40% weight on average distance from existing GREEN@ hubs.
        High score = large underserved population that is also far from current infrastructure.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SECTION 4 — MOBILE ROUTES
# ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<a name="mobile-route-simulation"></a>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">🚛 Mobile Route Simulation</h2>', unsafe_allow_html=True)
if True:
    st.markdown("""
    <div style='margin-bottom:12px'>
        <h3 style='margin:0'>🚛 Mobile Recycling Truck Routes</h3>
        <p style='color:#5a8f5a;font-size:0.84rem;margin:4px 0 0'>
            Simulated mobile collection trucks dispatched from the nearest GREEN@ hub
            to serve high-density estates beyond the service radius.
            Routing: greedy nearest-neighbour (TSP approximation).
        </p>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 3])
    with rc1:
        n_trucks = st.selectbox("Trucks dispatched", [2, 3, 4], index=1)
    with rc2:
        st.markdown(f"""
        <div style='padding-top:8px'>
            <span class='chip'>Real estate GPS</span>
            <span class='chip'>GREEN@ depots</span>
            <span class='chip'>{service_radius/1000:.1f}km threshold</span>
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("Computing optimised routes…"):
        route_map, us_estates = build_route_map(
            estates_df, rcp_df, green_df,
            service_radius_m=service_radius,
            n_trucks=n_trucks,
            green_dists=green_dists,
            road_adj=road_adj,
            road_coords=road_coords,
            road_grid=road_grid,
            road_grid_size=road_grid_size,
            composite_dists=composite_dists,
        )
        st_folium(route_map, width="100%", height=540, returned_objects=[])

    if not us_estates.empty:
        colors_css = ["#f0883e", "#79c0ff", "#d2a8ff", "#56d364"]
        names_list = ["Truck A", "Truck B", "Truck C", "Truck D"]
        pts = us_estates[["lat", "lng"]].values.tolist()
        cols_t = st.columns(n_trucks)

        for t in range(n_trucks):
            apt   = [pts[i] for i in range(len(pts)) if i % n_trucks == t]
            color = colors_css[t]
            tname = names_list[t]
            km = 0
            if len(apt) > 1:
                for i in range(len(apt) - 1):
                    km += haversine_m(apt[i][0], apt[i][1], apt[i+1][0], apt[i+1][1])
            km = round(km / 1000, 2)
            flats = int(sum(us_estates.iloc[i]["no_of_flats"]
                            for i in range(len(pts)) if i % n_trucks == t))
            with cols_t[t]:
                st.markdown(f"""
                <div style='background:#111e11;border:1px solid #1e3a1e;
                     border-left:4px solid {color};border-radius:10px;
                     padding:14px 16px;margin:4px 0'>
                    <div style='color:{color};font-weight:700;font-size:0.95rem'>{tname}</div>
                    <div style='color:#5a8f5a;font-size:0.8rem;margin-top:6px;line-height:1.9'>
                        Stops: <b style='color:#d4f0d4'>{len(apt)}</b><br>
                        Route: <b style='color:#d4f0d4'>{km} km</b><br>
                        Flats reached: <b style='color:#d4f0d4'>{flats:,}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("📋 Underserved estates detail"):
            show_cols = [c for c in ["estate_name", "district_name", "region_name",
                                      "no_of_flats", "dist_green_km"]
                         if c in us_estates.columns]
            tbl = us_estates[show_cols].copy()
            if "dist_green_km" in tbl.columns:
                tbl["dist_green_km"] = tbl["dist_green_km"].round(2)
            tbl.columns = ["Estate", "District", "Region",
                           "Flats", "Dist GREEN@ (km)"][:len(show_cols)]
            st.dataframe(tbl.sort_values("Dist GREEN@ (km)", ascending=False),
                         hide_index=True, use_container_width=True)
    else:
        st.info("No underserved estates at this radius. Reduce the service radius to see routes.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 5 — STRATEGY IMPACT DASHBOARD
# ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<a name="strategy-impact"></a>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">📊 Strategy Impact Dashboard</h2>', unsafe_allow_html=True)
if True:
    st.markdown("""
    <div style='margin-bottom:14px'>
        <h3 style='margin:0'>📊 Strategy Impact Dashboard</h3>
        <p style='color:#5a8f5a;font-size:0.84rem;margin:4px 0 0'>
            Quantify the expected outcome of the hybrid strategy
            (new fixed stations + mobile trucks) using real cost data
            and a participation uplift model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameter controls ────────────────────────────────────────────
    t5c1, t5c2, t5c3, t5c4 = st.columns(4)
    with t5c1:
        t5_n_stations = st.selectbox(
            "New GREEN@ stations", [0, 1, 3, 5], index=2, key="t5_nst")
    with t5c2:
        t5_n_trucks = st.selectbox(
            "Mobile trucks", [1, 2, 3, 4], index=1, key="t5_trucks")
    with t5c3:
        t5_threshold = st.slider(
            "Service radius (km)", 1.0, 5.0, 2.0, 0.5,
            key="t5_thresh")
    with t5c4:
        t5_part_factor = st.slider(
            "Participation uplift factor", 0.01, 0.06, 0.03, 0.01,
            key="t5_pf",
            help="Recycling participation gain per 500m distance reduction (default 3%)")

    t5_threshold_m = int(t5_threshold * 1000)

    # ── Recompute with user params ────────────────────────────────────
    gc5 = list(zip(green_df["lat"], green_df["lon"]))
    gd5 = np.array([min(haversine_m(lat, lng, g[0], g[1]) for g in gc5)
                    for lat, lng in zip(all_lats, all_lngs)])

    base5 = simulate_station_scenario(
        all_lats, all_lngs, all_flats, gc5, [], t5_threshold_m,
        small_coords=_sm_coords,
        walk_adj=walk_adj, walk_coords=walk_coords,
        walk_grid=walk_grid, walk_GRID=walk_GRID)

    # Get candidate clusters at this threshold (composite model)
    clust5, _ = compute_underserved_clusters(
        tuple(all_lats), tuple(all_lngs), tuple(all_flats),
        tuple(estates_df["estate_name"].tolist()),
        tuple(estates_df["district_name"].tolist()),
        tuple(green_df["lat"].tolist()),
        tuple(green_df["lon"].tolist()),
        small_lats=tuple(_small_premium["lat"].tolist()),
        small_lons=tuple(_small_premium["lon"].tolist()),
        threshold_m=t5_threshold_m,
        _walk_adj=walk_adj, _walk_coords=walk_coords,
        _walk_grid=walk_grid, _walk_GRID=walk_GRID,
    )
    top5_coords = list(zip(
        clust5.head(t5_n_stations)["centroid_lat"],
        clust5.head(t5_n_stations)["centroid_lon"],
    )) if t5_n_stations > 0 else []

    after5 = simulate_station_scenario(
        all_lats, all_lngs, all_flats, gc5, top5_coords, t5_threshold_m,
        small_coords=_sm_coords,
        walk_adj=walk_adj, walk_coords=walk_coords,
        walk_grid=walk_grid, walk_GRID=walk_GRID)

    newly5      = after5["flats_covered_2km"] - base5["flats_covered_2km"]
    dist_red5   = base5["avg_dist_km"] - after5["avg_dist_km"]

    cost5 = compute_cost_model(
        n_new_stations     = t5_n_stations,
        n_trucks           = t5_n_trucks,
        cost_data          = cost_data,
        us_flats           = after5["n_underserved"],
        total_flats        = sum(all_flats),
        baseline_rate      = 0.344,
        dist_red_km        = dist_red5,
        newly_covered_flats= newly5,
    )

    # ── Top metrics row ───────────────────────────────────────────────
    st.markdown("#### Projected outcomes")
    net = cost5["net_position"]
    st.markdown(f"""
    <div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:10px">
      <div style="background:#1a0a0a;border:1px solid #6e2020;border-left:3px solid #f85149;border-radius:10px;padding:14px 20px;flex:1;min-width:180px">
        <div style="color:#f85149;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;font-family:'DM Mono',monospace;margin-bottom:6px">Projected recycling rate</div>
        <div style="color:#ff8a80;font-size:2.2rem;font-weight:700;line-height:1">{cost5['proj_rate']}%</div>
        <div style="color:#c0504a;font-size:0.78rem;margin-top:4px">+{cost5['proj_rate']-34.4:.1f}pp vs 34.4% baseline</div>
      </div>
      <div style="background:#1a0a0a;border:1px solid #6e2020;border-left:3px solid #f85149;border-radius:10px;padding:14px 20px;flex:1;min-width:180px">
        <div style="color:#f85149;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;font-family:'DM Mono',monospace;margin-bottom:6px">Extra flats recycling</div>
        <div style="color:#ff8a80;font-size:2.2rem;font-weight:700;line-height:1">{cost5['extra_flats']:,}</div>
        <div style="color:#c0504a;font-size:0.78rem;margin-top:4px">newly active recyclers</div>
      </div>
      <div style="background:#1a0a0a;border:1px solid #6e2020;border-left:3px solid #f85149;border-radius:10px;padding:14px 20px;flex:1;min-width:180px">
        <div style="color:#f85149;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;font-family:'DM Mono',monospace;margin-bottom:6px">Extra recovered (t/yr)</div>
        <div style="color:#ff8a80;font-size:2.2rem;font-weight:700;line-height:1">{cost5['extra_tonnes_yr']:,}</div>
      </div>
    </div>
    <div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:10px">
      <div style="background:#0a1a0a;border:1px solid #1e4a1e;border-left:3px solid #3fb950;border-radius:10px;padding:14px 20px;flex:1;min-width:240px">
        <div style="color:#3fb950;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;font-family:'DM Mono',monospace;margin-bottom:6px">Financial benefit (HKD/yr)</div>
        <div style="color:#56d364;font-size:1.9rem;font-weight:700;line-height:1">{cost5['financial_benefit']:,.0f}</div>
        <div style="color:#4a8f4a;font-size:0.78rem;margin-top:4px">avoided disposal + material value</div>
      </div>
      <div style="background:#0a1a0a;border:1px solid #1e4a1e;border-left:3px solid #3fb950;border-radius:10px;padding:14px 20px;flex:1;min-width:240px">
        <div style="color:#3fb950;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;font-family:'DM Mono',monospace;margin-bottom:6px">Net position (HKD/yr)</div>
        <div style="color:#56d364;font-size:1.9rem;font-weight:700;line-height:1">{net:+,.0f}</div>
        <div style="color:#4a8f4a;font-size:0.78rem;margin-top:4px">benefit minus operating cost</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Cost breakdown + impact side by side ─────────────────────────
    cost_col, impact_col = st.columns(2)

    with cost_col:
        st.markdown("#### Annual cost breakdown")
        if cost_data:
            truck_d = cost_data["recycle_truck_cost"][1]
            sta_d   = cost_data["recycle_station_cost"][0]
            truck_ann = sum(truck_d["annual_cost_per_truck"]) / 2
            sta_ann   = sum(sta_d["annual_cost_per_station"]) / 2
            st.markdown(f"""
            <div class='fact-card'>
                <b>Cost data source:</b> {cost_data['data_info']['source']}<br>
                <b>Reference year:</b> {cost_data['data_info']['year']}
            </div>
            """, unsafe_allow_html=True)
            cost_rows = [
                ["Hybrid truck (annual/vehicle)",
                 f"HKD {truck_ann:,.0f}",
                 f"×{t5_n_trucks} = HKD {t5_n_trucks*truck_ann:,.0f}"],
                [f"Large GREEN@ station (annual)",
                 f"HKD {sta_ann:,.0f}",
                 f"×{t5_n_stations} = HKD {t5_n_stations*sta_ann:,.0f}"],
                ["Cost per tonne (public)",
                 f"HKD {cost_data['summary_data']['public_avg_cost_per_ton']:,}", ""],
                ["Cost per tonne (private)",
                 f"HKD {cost_data['summary_data']['private_avg_cost_per_ton']:,}", ""],
                ["TOTAL annual operating cost", "",
                 f"HKD {cost5['total_annual_cost']:,.0f}"],
            ]
        else:
            cost_rows = [
                ["Mobile trucks", "", f"HKD {cost5['annual_mobile_cost']:,.0f}"],
                ["Fixed stations", "", f"HKD {cost5['annual_station_cost']:,.0f}"],
                ["Total", "", f"HKD {cost5['total_annual_cost']:,.0f}"],
            ]
        cdf = pd.DataFrame(cost_rows, columns=["Item", "Unit cost", "Total"])
        st.dataframe(cdf, hide_index=True, use_container_width=True)

        # Stacked cost bar
        cost_bar = pd.DataFrame({
            "Mobile trucks":  [cost5["annual_mobile_cost"]],
            "Fixed stations": [cost5["annual_station_cost"]],
        })
        st.bar_chart(cost_bar, color=["#79c0ff", "#3fb950"], height=160)

    with impact_col:
        st.markdown("#### Coverage vs baseline")

        # Before/after comparison table
        comp5 = pd.DataFrame({
            "Metric": [
                "Estates within radius (%)",
                "Avg dist to GREEN@ (km)",
                "Underserved estates",
                "Flats within radius",
                "Weighted coverage score",
                "Recycling participation",
            ],
            "Baseline": [
                f"{base5['pct_covered_2km']}%",
                f"{base5['avg_dist_km']} km",
                str(base5["n_underserved"]),
                f"{base5['flats_covered_2km']:,}",
                f"{base5['weighted_score']}",
                "34.4%",
            ],
            "After strategy": [
                f"{after5['pct_covered_2km']}%",
                f"{after5['avg_dist_km']} km",
                str(after5["n_underserved"]),
                f"{after5['flats_covered_2km']:,}",
                f"{after5['weighted_score']}",
                f"{cost5['proj_rate']}%",
            ],
            "Change": [
                f"+{after5['pct_covered_2km']-base5['pct_covered_2km']:.1f}pp",
                f"{after5['avg_dist_km']-base5['avg_dist_km']:+.2f} km",
                f"{after5['n_underserved']-base5['n_underserved']:+d}",
                f"+{after5['flats_covered_2km']-base5['flats_covered_2km']:,}",
                f"+{after5['weighted_score']-base5['weighted_score']:.3f}",
                f"+{cost5['proj_rate']-34.4:.1f}pp",
            ],
        })
        st.dataframe(comp5, hide_index=True, use_container_width=True, height=260)

    # ── Participation model chart ──────────────────────────────────────
    st.markdown("#### Scenario sweep — participation rate vs number of new stations")
    sweep_rows = []
    for ns in [0, 1, 2, 3, 5, 8]:
        top_ns = clust5.head(ns) if ns > 0 else pd.DataFrame()
        nc = list(zip(top_ns["centroid_lat"], top_ns["centroid_lon"])) if not top_ns.empty else []
        a  = simulate_station_scenario(all_lats, all_lngs, all_flats, gc5, nc, t5_threshold_m,
                                       small_coords=_sm_coords,
                                       walk_adj=walk_adj, walk_coords=walk_coords,
                                       walk_grid=walk_grid, walk_GRID=walk_GRID)
        nl = a["flats_covered_2km"] - base5["flats_covered_2km"]
        dr = base5["avg_dist_km"] - a["avg_dist_km"]
        cm = compute_cost_model(ns, t5_n_trucks, cost_data,
                                a["n_underserved"], sum(all_flats),
                                0.344, dr, nl)
        sweep_rows.append({
            "New stations":        ns,
            "Participation (%)":   cm["proj_rate"],
            "Flats covered (%)":   a["pct_covered_2km"],
            "Underserved":         a["n_underserved"],
            "Total cost (M HKD)":  round(cm["total_annual_cost"] / 1e6, 2),
        })
    sweep_df = pd.DataFrame(sweep_rows).set_index("New stations")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Participation rate by scenario**")
        try:
            import plotly.graph_objects as go
            y_vals = sweep_df["Participation (%)"].tolist()
            y_min = max(0, min(y_vals) - 0.3)
            y_max = max(y_vals) + 0.3
            fig_part = go.Figure()
            fig_part.add_trace(go.Scatter(
                x=sweep_df.index.tolist(), y=y_vals,
                mode="lines+markers",
                line=dict(color="#3fb950", width=2),
                marker=dict(size=7),
            ))
            fig_part.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#5a8f5a", title="New stations", gridcolor="#1e3a1e"),
                yaxis=dict(color="#5a8f5a", range=[y_min, y_max],
                           ticksuffix="%", gridcolor="#1e3a1e"),
                font=dict(color="#5a8f5a"),
            )
            st.plotly_chart(fig_part, use_container_width=True)
        except ImportError:
            st.line_chart(sweep_df[["Participation (%)"]], color=["#3fb950"], height=220)
    with sc2:
        st.markdown("**Flats covered (%) vs operating cost (M HKD)**")
        st.line_chart(sweep_df[["Flats covered (%)", "Total cost (M HKD)"]],
                      color=["#79c0ff", "#f85149"], height=220)

    # ── Cost structure from JSON ───────────────────────────────────────
    if cost_data:
        st.markdown("#### Station type cost comparison (from cost data)")
        sta_rows = []
        for s in cost_data["recycle_station_cost"]:
            ann = s["annual_cost_per_station"]
            if isinstance(ann, list):
                ann_str = f"HKD {ann[0]:,} – {ann[1]:,}"
            elif ann:
                ann_str = f"HKD {ann:,}"
            else:
                ann_str = "Private / variable"
            vol = s["daily_recycle_volume_t"]
            vol_str = f"{vol[0]}–{vol[1]} t/day" if isinstance(vol, list)                       else (f"{vol} t/day" if vol else "—")
            cpt = s["cost_per_ton"]
            cpt_str = f"HKD {cpt:,}" if isinstance(cpt, (int,float)) else str(cpt)
            sta_rows.append({
                "Type":          s["station_type"],
                "Annual cost":   ann_str,
                "Daily volume":  vol_str,
                "Cost/tonne":    cpt_str,
                "Labour %":      f"{int(s['cost_ratio']['labor']*100)}%",
                "Rent %":        f"{int(s['cost_ratio']['rent']*100)}%",
                "Transport %":   f"{int(s['cost_ratio']['transport']*100)}%",
            })
        st.dataframe(pd.DataFrame(sta_rows), hide_index=True, use_container_width=True)

        st.markdown("#### Truck type cost comparison")
        trk_rows = []
        for t in cost_data["recycle_truck_cost"]:
            ann = t["annual_cost_per_truck"]
            ann_str = f"HKD {ann[0]:,}–{ann[1]:,}" if isinstance(ann,list)                       else ("Private" if not ann else f"HKD {ann:,}")
            tpt = t["transport_cost_per_ton"]
            tpt_str = f"HKD {tpt[0]:,}–{tpt[1]:,}/t" if isinstance(tpt,list)                       else f"HKD {tpt:,}/t"
            trk_rows.append({
                "Type":            t["truck_type"],
                "Annual cost":     ann_str,
                "Transport cost":  tpt_str,
                "Monthly km":      t["monthly_mileage_km"] or "—",
            })
        st.dataframe(pd.DataFrame(trk_rows), hide_index=True, use_container_width=True)

    st.markdown("""
    <div class='fact-card'>
        <b>Model assumptions:</b>
        Participation uplift = 3% per 500m distance reduction + 10% for full-service station access.
        Financial benefit = (avoided landfill HKD 450/t) + (recovered material HKD 200/t).
        Cost data sourced from Hong Kong EPD, Green@Community, and Recycling Industry General Chamber of Commerce.
    </div>
    """, unsafe_allow_html=True)
