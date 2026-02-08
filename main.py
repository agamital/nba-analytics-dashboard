"""
NBA Analytics Dashboard

A Streamlit application for analyzing NBA statistics, teams, and players.
Features standings, team analysis, player explorer, and comparison tools.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from nba_client import (
    get_standings,
    get_teams_df,
    get_team_abbr,
    get_team_logo_url_by_abbr,
    get_team_roster,
    get_team_roster_player_ids,
    get_team_last_games,
    search_players,
    get_player_game_log,
    summarize_player,
    compute_player_splits,
    synergy_matrix,
    group_summary,
    group_daily_series,
    recent_vs_season,
    add_rolling_metrics,
    add_derived_player_columns,
    filter_log_by_date,
    common_date_window,
    calculate_per,
)

# Configuration Constants (embedded directly)
SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23"]
HEADSHOT_WIDTH = 95
TEAM_LOGO_WIDTH = 70
PLAYER_BASIC_METRICS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
TEAM_METRICS = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "FG3M", "REB", "AST", "TOV", "STL", "BLK"]
DEFAULT_LAST_N_GAMES = 20
DEFAULT_ROLLING_WINDOWS = (5, 10)


# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="NBA Analytics Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏀 NBA Analytics Dashboard")
st.caption("Standings • Team Analysis • Player Explorer • Compare & Synergy")

# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------
page = st.sidebar.radio(
    "📊 Navigation",
    ["Standings", "Team", "Player Explorer", "Compare & Synergy"],
    index=0,
)
season = st.sidebar.selectbox("🏆 Season", SEASONS, index=0)

with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown("""
    **NBA Analytics Dashboard**
    
    This app uses the `nba_api` library to fetch real-time NBA statistics.
    
    **Features:**
    - Live standings with filters
    - Team analysis & comparison
    - Player statistics & trends
    - Advanced synergy metrics
    
    **Note:** If you encounter timeouts or rate limits, wait 30-60 seconds and refresh.
    Data is cached for 6 hours to improve performance.
    """)

with st.sidebar.expander("🛠️ Settings", expanded=False):
    show_debug = st.checkbox("Show debug info", value=False)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def throttle(seconds: float = 0.6):
    """Sleep to avoid API rate limits."""
    time.sleep(seconds)


def player_headshot_url(player_id: int) -> str:
    """Generate player headshot URL."""
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"


def show_error(message: str, exception: Exception = None):
    """Display user-friendly error message."""
    st.error(f"⚠️ {message}")
    if exception and show_debug:
        st.exception(exception)


def metric_row(summary: pd.Series):
    """Display player statistics in a grid layout."""
    # Basic stats
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("Games", int(summary.get("Games", 0)))
    a2.metric("PTS", f"{summary.get('PTS', 0):.1f}")
    a3.metric("REB", f"{summary.get('REB', 0):.1f}")
    a4.metric("AST", f"{summary.get('AST', 0):.1f}")
    a5.metric("STL", f"{summary.get('STL', 0):.1f}")
    a6.metric("BLK", f"{summary.get('BLK', 0):.1f}")

    # Shooting stats
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("FG%", f"{summary.get('FG_PCT', 0):.3f}" if "FG_PCT" in summary else "—")
    b2.metric("3P%", f"{summary.get('FG3_PCT', 0):.3f}" if "FG3_PCT" in summary else "—")
    b3.metric("FT%", f"{summary.get('FT_PCT', 0):.3f}" if "FT_PCT" in summary else "—")
    b4.metric("3PM", f"{summary.get('FG3M', 0):.1f}" if "FG3M" in summary else "—")
    b5.metric("TOV", f"{summary.get('TOV', 0):.1f}")
    b6.metric("+/-", f"{summary.get('PLUS_MINUS', 0):.1f}" if "PLUS_MINUS" in summary else "—")


# --------------------------------------------------
# State Management
# --------------------------------------------------
state_key = f"{page}:{season}"
if st.session_state.get("_state_key") != state_key:
    st.session_state["_state_key"] = state_key
    st.session_state.pop("pool", None)
    st.session_state.pop("_last_compare_logs", None)


# --------------------------------------------------
# Cached Data Loaders
# --------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, persist="disk", show_spinner="Loading standings...")
def load_standings(season: str) -> pd.DataFrame:
    """Load NBA standings with caching."""
    return get_standings(season)


@st.cache_data(ttl=6 * 60 * 60, persist="disk")
def load_teams_static() -> pd.DataFrame:
    """Load static team data with caching."""
    return get_teams_df()


@st.cache_data(ttl=6 * 60 * 60, persist="disk", show_spinner="Loading roster...")
def load_roster(team_id: int, season: str) -> pd.DataFrame:
    """Load team roster with caching."""
    return get_team_roster(team_id, season)


@st.cache_data(ttl=6 * 60 * 60, persist="disk")
def load_roster_ids(team_id: int, season: str) -> pd.DataFrame:
    """Load roster player IDs with caching."""
    return get_team_roster_player_ids(team_id, season)


@st.cache_data(ttl=6 * 60 * 60, persist="disk", show_spinner="Loading games...")
def load_team_games(team_id: int, season: str, last_n: int = None) -> pd.DataFrame:
    """Load team game log with caching."""
    return get_team_last_games(team_id, season, last_n)


@st.cache_data(ttl=6 * 60 * 60, persist="disk", show_spinner="Loading player stats...")
def load_player_log(player_id: int, season: str) -> pd.DataFrame:
    """Load player game log with caching."""
    return get_player_game_log(player_id, season)


@st.cache_data(ttl=6 * 60 * 60, persist="disk")
def load_team_abbr(team_id: int) -> str:
    """Load team abbreviation with caching."""
    return get_team_abbr(team_id)


# ==================================================
# PAGE: STANDINGS
# ==================================================
if page == "Standings":
    st.subheader("📊 NBA Standings")

    try:
        df = load_standings(season)
    except Exception as e:
        show_error(
            "Unable to load standings. The NBA API may be experiencing issues. "
            "Please wait 30-60 seconds and try again.",
            e
        )
        st.stop()

    # Filters
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        conf = st.selectbox("Conference", ["All", "East", "West"])
    with c2:
        min_wins = st.number_input("Min wins", 0, 82, 0)
    with c3:
        search = st.text_input("Search team", placeholder="Enter team name...")

    # Apply filters
    out = df.copy()
    if conf != "All":
        out = out[out["Conference"] == conf]
    out = out[out["WINS"] >= min_wins]
    if search.strip():
        out = out[out["TeamName"].str.contains(search, case=False, na=False)]

    # Display in tabs
    tab1, tab2 = st.tabs(["🏀 Eastern Conference", "🏀 Western Conference"])
    with tab1:
        east = out[out["Conference"] == "East"]
        if east.empty:
            st.info("No teams match the filter criteria.")
        else:
            st.dataframe(east, use_container_width=True, hide_index=True)
    with tab2:
        west = out[out["Conference"] == "West"]
        if west.empty:
            st.info("No teams match the filter criteria.")
        else:
            st.dataframe(west, use_container_width=True, hide_index=True)


# ==================================================
# PAGE: TEAM
# ==================================================
elif page == "Team":
    st.subheader("🏀 Team Analysis")

    teams_df = load_teams_static()
    mode = st.radio("Mode", ["Single Team Analysis", "Compare Teams"], horizontal=True)

    # ==================================================
    # SINGLE TEAM
    # ==================================================
    if mode == "Single Team Analysis":
        last_n = st.slider("Last N games", 5, 82, DEFAULT_LAST_N_GAMES, key="team_single_last_n")

        team_full = st.selectbox("Select team", teams_df["full_name"].tolist())
        team_id = int(teams_df.loc[teams_df["full_name"] == team_full, "id"].iloc[0])

        # Team logo
        logo_url = None
        try:
            abbr = load_team_abbr(team_id)
            logo_url = get_team_logo_url_by_abbr(abbr)
        except Exception:
            pass

        c_logo, c_title = st.columns([1, 7])
        with c_logo:
            if logo_url:
                st.image(logo_url, width=TEAM_LOGO_WIDTH)
        with c_title:
            st.markdown(f"## {team_full}")

        # Roster
        try:
            roster = load_roster(team_id, season)
        except Exception as e:
            show_error("Unable to load team roster. Please try again later.", e)
            st.stop()

        st.markdown("### 👥 Roster")
        st.dataframe(roster, use_container_width=True, hide_index=True)

        # Recent games
        try:
            games = load_team_games(team_id, season, last_n)
        except Exception as e:
            show_error("Unable to load team game log. Please try again later.", e)
            st.stop()

        st.markdown(f"### 📅 Last {last_n} Games")
        st.dataframe(games, use_container_width=True, hide_index=True)

        # Trends
        if "GAME_DATE" in games.columns and not games.empty:
            g = games.copy()
            g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"], errors="coerce")
            g = g.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").set_index("GAME_DATE")

            chart_cols = [c for c in ["PTS", "FG3M", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV"] 
                         if c in g.columns]
            if chart_cols:
                st.markdown("### 📈 Team Trends")
                for c in chart_cols:
                    st.line_chart(g[[c]], use_container_width=True)

    # ==================================================
    # COMPARE TEAMS
    # ==================================================
    else:
        picks = st.multiselect(
            "Select teams to compare",
            teams_df["full_name"].tolist(),
            default=teams_df["full_name"].tolist()[:2] if len(teams_df) >= 2 else [],
        )
        
        if len(picks) < 2:
            st.info("📌 Please select at least 2 teams to compare.")
            st.stop()

        metric_picks = st.multiselect(
            "Metrics to compare (select 1-3 for best visibility)",
            TEAM_METRICS,
            default=["PTS", "FG3M"],
        )
        
        if not metric_picks:
            st.info("📌 Please select at least 1 metric.")
            st.stop()

        # Filter mode
        filter_mode = st.radio(
            "Filter games by",
            ["Last N games", "Date range"],
            horizontal=True,
            key="team_compare_filter_mode",
        )

        # Fetch full season logs for all teams
        logs = {}
        for team_full in picks:
            team_id = int(teams_df.loc[teams_df["full_name"] == team_full, "id"].iloc[0])
            try:
                df = load_team_games(team_id, season, None)  # Full season
            except Exception as e:
                show_error(f"Failed to load data for {team_full}.", e)
                st.stop()

            if df is None or df.empty:
                continue

            # Ensure GAME_DATE is datetime
            if "GAME_DATE" in df.columns:
                df = df.copy()
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
                df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")
                df = df.set_index("GAME_DATE")

            logs[team_full] = df

        if not logs:
            show_error("No team data returned. Please try again.")
            st.stop()

        # Compute date range
        all_dates = []
        for dfx in logs.values():
            if dfx is None or dfx.empty:
                continue
            if isinstance(dfx.index, pd.DatetimeIndex):
                all_dates.extend(dfx.index.dropna().tolist())

        if not all_dates:
            show_error("No dates found in team game logs.")
            st.stop()

        min_date = min(all_dates).date()
        max_date = max(all_dates).date()

        # Apply filters
        if filter_mode == "Date range":
            c1, c2 = st.columns(2)
            with c1:
                start_date = st.date_input(
                    "Start date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="team_cmp_start",
                )
            with c2:
                end_date = st.date_input(
                    "End date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="team_cmp_end",
                )

            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            logs = {
                name: dfx[(dfx.index >= start_ts) & (dfx.index <= end_ts)]
                for name, dfx in logs.items()
                if dfx is not None and not dfx.empty
            }
        else:
            # Last N games
            n = st.slider("Last N games", 5, 82, DEFAULT_LAST_N_GAMES, key="team_compare_last_n")
            logs = {
                name: dfx.sort_index(ascending=False).head(n).sort_index()
                for name, dfx in logs.items()
                if dfx is not None and not dfx.empty
            }

        # Remove empty logs
        logs = {k: v for k, v in logs.items() if v is not None and not v.empty}
        if not logs:
            st.warning("⚠️ After filtering, no games remain for the selected teams.")
            st.stop()

        # Trend charts
        st.markdown("### 📈 Trends Over Time")

        for m in metric_picks[:3]:  # Limit to 3 metrics
            long_rows = []
            for team_full, dfx in logs.items():
                if dfx is None or dfx.empty or m not in dfx.columns:
                    continue
                tmp = dfx[[m]].copy()
                tmp[m] = pd.to_numeric(tmp[m], errors="coerce")
                tmp = tmp.dropna().reset_index()
                tmp["Team"] = team_full
                tmp.rename(columns={m: "Value"}, inplace=True)
                long_rows.append(tmp[["GAME_DATE", "Team", "Value"]])

            if not long_rows:
                st.info(f"No data available for {m}.")
                continue

            long_df = pd.concat(long_rows, ignore_index=True)
            long_df["GAME_DATE"] = pd.to_datetime(long_df["GAME_DATE"], errors="coerce")

            chart = (
                alt.Chart(long_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("GAME_DATE:T", title="Game Date"),
                    y=alt.Y("Value:Q", title=m),
                    color=alt.Color("Team:N", legend=alt.Legend(title="Team")),
                    tooltip=[
                        "Team:N",
                        alt.Tooltip("GAME_DATE:T", title="Date"),
                        alt.Tooltip("Value:Q", format=".2f", title=m)
                    ],
                )
                .properties(height=320)
                .interactive()
            )

            st.markdown(f"#### {m}")
            st.altair_chart(chart, use_container_width=True)

        # Averages bar chart
        st.markdown("### 📊 Averages (Selected Window)")
        avg_rows = []
        for team_full, dfx in logs.items():
            row = {"Team": team_full}
            for m in metric_picks[:5]:
                if m in dfx.columns:
                    row[m] = pd.to_numeric(dfx[m], errors="coerce").mean()
            avg_rows.append(row)

        avg_df = pd.DataFrame(avg_rows).set_index("Team")
        st.bar_chart(avg_df, use_container_width=True)


# ==================================================
# PAGE: PLAYER EXPLORER
# ==================================================
elif page == "Player Explorer":
    st.subheader("👤 Player Explorer")

    mode = st.radio("Find player by", ["Global Search", "Team Roster"], horizontal=True)

    player_name = None
    player_id = None
    team_logo_url = None

    # Global Search
    if mode == "Global Search":
        with st.form("player_search_form"):
            q = st.text_input("Search player", placeholder="e.g., LeBron, Curry, Doncic")
            submitted = st.form_submit_button("🔍 Search")

        if not submitted or not q.strip():
            st.info("💡 Enter a player name and click Search.")
            st.stop()

        try:
            results = search_players(q.strip())
        except Exception as e:
            show_error("Search failed. Please try again.", e)
            st.stop()

        if results.empty:
            st.warning("No players found matching your search.")
            st.stop()

        st.dataframe(results, use_container_width=True, hide_index=True)
        player_name = st.selectbox("Select player", results["full_name"].tolist())
        player_id = int(results.loc[results["full_name"] == player_name, "id"].iloc[0])

    # Team Roster
    else:
        teams_df = load_teams_static()
        team_full = st.selectbox("Select team", teams_df["full_name"].tolist(), key="px_team")
        team_id = int(teams_df.loc[teams_df["full_name"] == team_full, "id"].iloc[0])

        try:
            abbr = load_team_abbr(team_id)
            team_logo_url = get_team_logo_url_by_abbr(abbr)
        except Exception:
            team_logo_url = None

        if team_logo_url:
            st.image(team_logo_url, width=TEAM_LOGO_WIDTH)

        try:
            roster_ids = load_roster_ids(team_id, season)
        except Exception as e:
            show_error("Unable to load roster. Please try again.", e)
            st.stop()

        if roster_ids.empty:
            st.warning("Roster is empty for this season/team.")
            st.stop()

        player_name = st.selectbox("Pick player", roster_ids["PLAYER"].tolist())
        player_id = int(roster_ids.loc[roster_ids["PLAYER"] == player_name, "PLAYER_ID"].iloc[0])

    # Load player game log
    try:
        log = load_player_log(player_id, season)
    except Exception as e:
        show_error(
            "Unable to load player data. The NBA API may be experiencing high traffic. "
            "Please wait 30-60 seconds and try again.",
            e
        )
        st.stop()

    if log.empty:
        st.warning("No games found for this player in the selected season.")
        st.stop()

    log = add_derived_player_columns(log)

    # Player header
    c1, c2, c3 = st.columns([1, 6, 2])
    with c1:
        st.image(player_headshot_url(player_id), width=HEADSHOT_WIDTH)
    with c2:
        st.markdown(f"## {player_name}")
    with c3:
        if team_logo_url:
            st.image(team_logo_url, width=TEAM_LOGO_WIDTH)

    # Season summary
    summary = summarize_player(log)
    metric_row(summary)

    # Tabs for different views
    tabs = st.tabs(["📊 Overview", "🔀 Splits", "📈 Trends", "📋 Game Log"])

    with tabs[0]:
        st.markdown("### Recent Form vs Season Average")
        comp = recent_vs_season(log, last_n=10)
        if not comp.empty:
            st.dataframe(comp, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for comparison.")

        st.markdown("### Key Statistics Over Time")
        if "GAME_DATE" in log.columns:
            base_cols = [c for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", 
                                     "FG_PCT", "FG3_PCT", "FT_PCT", "FG3M"]
                         if c in log.columns]
            if base_cols:
                st.line_chart(log.set_index("GAME_DATE")[base_cols], use_container_width=True)

    with tabs[1]:
        st.markdown("### Performance Splits (Home/Away, Win/Loss)")
        splits = compute_player_splits(log)
        if splits.empty:
            st.info("Not enough data to calculate splits.")
        else:
            st.dataframe(splits, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("### Rolling Averages")
        metrics = [c for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", 
                              "FG_PCT", "FG3_PCT", "FT_PCT", "FG3M"]
                   if c in log.columns]
        roll = add_rolling_metrics(log, metrics=metrics, windows=DEFAULT_ROLLING_WINDOWS)
        
        if "GAME_DATE" in roll.columns:
            roll = roll.set_index("GAME_DATE")
            roll_cols = [c for c in roll.columns if c.endswith(("roll5", "roll10"))]
            if roll_cols:
                st.line_chart(roll[roll_cols], use_container_width=True)

        st.markdown("### Shooting Volume (Attempts)")
        vol_cols = [c for c in ["FGA", "FG3A", "FTA"] if c in log.columns]
        if vol_cols:
            st.line_chart(log.set_index("GAME_DATE")[vol_cols], use_container_width=True)

    with tabs[3]:
        st.dataframe(log, use_container_width=True, hide_index=True)


# ==================================================
# PAGE: COMPARE & SYNERGY
# ==================================================
else:
    st.subheader("🔄 Compare & Synergy")

    t_players, t_teams = st.tabs(["👥 Players Compare", "🏀 Teams Compare"])

    # =========================
    # Players Compare
    # =========================
    with t_players:
        st.write("Add players to a pool, organize into groups, filter by date, and analyze synergy.")

        if "pool" not in st.session_state:
            st.session_state.pool = {}  # name -> id
        
        if "search_results" not in st.session_state:
            st.session_state.search_results = None

        add_mode = st.radio(
            "Add players by", 
            ["Global Search", "Team Roster"], 
            horizontal=True, 
            key="cmp_add_mode"
        )

        # Add players - Global Search
        if add_mode == "Global Search":
            with st.form("add_player_global_form"):
                q = st.text_input("Search player to add", key="cmp_q", placeholder="Enter player name")
                do_search = st.form_submit_button("🔍 Search")

            if do_search and q.strip():
                try:
                    st.session_state.search_results = search_players(q.strip())
                except Exception as e:
                    show_error("Search failed.", e)
                    st.session_state.search_results = None

            # Display search results if available
            if st.session_state.search_results is not None and not st.session_state.search_results.empty:
                res = st.session_state.search_results
                st.dataframe(res, use_container_width=True, hide_index=True)
                
                # Show available players to add
                available_players = [p for p in res["full_name"].tolist() if p not in st.session_state.pool]
                
                if not available_players:
                    st.info("✅ All players from this search are already in the pool!")
                else:
                    st.markdown("**Select players to add:**")
                    for idx, player_name in enumerate(available_players):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"🏀 {player_name}")
                        with col2:
                            if st.button("➕ Add", key=f"add_player_{idx}_{player_name.replace(' ', '_')}"):
                                pid = int(res.loc[res["full_name"] == player_name, "id"].iloc[0])
                                st.session_state.pool[player_name] = pid
                                st.success(f"✅ Added {player_name} to pool!")
                                time.sleep(0.5)  # Brief pause so user sees success message
                                st.rerun()
            elif do_search and st.session_state.search_results is not None and st.session_state.search_results.empty:
                st.warning("No players found matching your search.")

        # Add players - Team Roster
        else:
            teams_df = load_teams_static()
            team_full = st.selectbox("Team", teams_df["full_name"].tolist(), key="cmp_team")
            team_id = int(teams_df.loc[teams_df["full_name"] == team_full, "id"].iloc[0])

            try:
                abbr = load_team_abbr(team_id)
                st.image(get_team_logo_url_by_abbr(abbr), width=60)
            except Exception:
                pass

            try:
                roster_ids = load_roster_ids(team_id, season)
            except Exception as e:
                show_error("Roster fetch failed.", e)
                st.stop()

            if roster_ids.empty:
                st.warning("Roster is empty.")
            else:
                picks = st.multiselect(
                    "Select players to add to pool", 
                    roster_ids["PLAYER"].tolist(), 
                    key="cmp_roster_multi"
                )
                for p in picks:
                    pid = int(roster_ids.loc[roster_ids["PLAYER"] == p, "PLAYER_ID"].iloc[0])
                    st.session_state.pool[p] = pid

        st.divider()

        # Show pool
        if not st.session_state.pool:
            st.info("💡 The pool is empty. Add players using the options above.")
            st.stop()

        st.markdown("### 👥 Current Pool")
        
        # Fetch PER ratings for pool players
        pool_data = []
        for name, pid in st.session_state.pool.items():
            try:
                log = load_player_log(pid, season)
                per = calculate_per(log)
                avg_pts = summarize_player(log).get('PTS', 0)
                pool_data.append({
                    'name': name,
                    'pid': pid,
                    'per': per,
                    'pts': avg_pts
                })
            except:
                pool_data.append({
                    'name': name,
                    'pid': pid,
                    'per': 0,
                    'pts': 0
                })
        
        # Sort by PER
        pool_data.sort(key=lambda x: x['per'], reverse=True)
        
        # Display in grid with images
        cols_per_row = 4
        for i in range(0, len(pool_data), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(pool_data):
                    player = pool_data[i + j]
                    with col:
                        st.image(player_headshot_url(player['pid']), use_container_width=True)
                        st.markdown(f"**{player['name']}**")
                        st.caption(f"⭐ PER: {player['per']:.1f}")
                        st.caption(f"📊 {player['pts']:.1f} PPG")
        
        # Clear pool button
        if st.button("🗑️ Clear Entire Pool"):
            st.session_state.pool = {}
            st.rerun()

        # Select groups
        names = sorted(list(st.session_state.pool.keys()))
        st.markdown("### 🔀 Select Groups to Compare")
        ca, cb = st.columns(2)
        with ca:
            group_a = st.multiselect(
                "Group A", 
                names, 
                default=names[:min(3, len(names))], 
                key="ga"
            )
        with cb:
            default_b = names[3:min(6, len(names))]
            group_b = st.multiselect("Group B", names, default=default_b, key="gb")

        # Remove duplicates
        group_b = [p for p in group_b if p not in set(group_a)]

        if not group_a or not group_b:
            st.info("💡 Select at least 1 player in each group to compare.")
            st.stop()

        # Display headshots
        st.markdown("### 👤 Group Members")
        ha, hb = st.columns(2)
        with ha:
            st.markdown("#### Group A")
            cols = st.columns(4)
            for i, name in enumerate(group_a):
                pid = st.session_state.pool[name]
                with cols[i % 4]:
                    st.image(player_headshot_url(pid), use_container_width=True)
                    st.caption(name)
        with hb:
            st.markdown("#### Group B")
            cols = st.columns(4)
            for i, name in enumerate(group_b):
                pid = st.session_state.pool[name]
                with cols[i % 4]:
                    st.image(player_headshot_url(pid), use_container_width=True)
                    st.caption(name)

        # Fetch logs
        try:
            with st.spinner("Fetching player game logs..."):
                logs_a = {}
                for n in group_a:
                    logs_a[n] = add_derived_player_columns(load_player_log(st.session_state.pool[n], season))
                    throttle()

                logs_b = {}
                for n in group_b:
                    logs_b[n] = add_derived_player_columns(load_player_log(st.session_state.pool[n], season))
                    throttle()
        except Exception as e:
            show_error(
                "Failed to fetch player logs. The API may be rate-limited. "
                "Please wait 30-60 seconds and try again.",
                e
            )
            st.stop()

        # Date filtering
        all_logs_raw = {**logs_a, **logs_b}
        min_d, max_d = common_date_window(all_logs_raw)
        
        if min_d is None:
            show_error("No dates found in player game logs.")
            st.stop()

        st.markdown("## 🗓️ Date Filters")
        f1, f2, f3 = st.columns([2, 2, 2])
        with f1:
            start_date = st.date_input(
                "Start date", 
                value=min_d.date(), 
                min_value=min_d.date(), 
                max_value=max_d.date()
            )
        with f2:
            end_date = st.date_input(
                "End date", 
                value=max_d.date(), 
                min_value=min_d.date(), 
                max_value=max_d.date()
            )
        with f3:
            preset = st.selectbox(
                "Quick preset", 
                ["Custom", "Last 7 games", "Last 14 games", "Last 30 games"], 
                index=0
            )

        # Apply preset
        if preset != "Custom":
            n = {"Last 7 games": 7, "Last 14 games": 14, "Last 30 games": 30}[preset]
            all_dates = []
            for df in all_logs_raw.values():
                if df is None or df.empty or "GAME_DATE" not in df.columns:
                    continue
                d = pd.to_datetime(df["GAME_DATE"], errors="coerce").dropna().dt.normalize().unique().tolist()
                all_dates.extend(d)
            if all_dates:
                all_dates = sorted(pd.to_datetime(pd.Series(all_dates)).unique())
                end_date = pd.to_datetime(all_dates[-1]).date()
                start_date = pd.to_datetime(all_dates[max(0, len(all_dates) - n)]).date()
                st.caption(f"✅ Preset applied: {preset} → {start_date} to {end_date}")

        # Filter logs
        logs_a = {k: filter_log_by_date(v, start_date, end_date) for k, v in logs_a.items()}
        logs_b = {k: filter_log_by_date(v, start_date, end_date) for k, v in logs_b.items()}

        logs_a = {k: v for k, v in logs_a.items() if v is not None and not v.empty}
        logs_b = {k: v for k, v in logs_b.items() if v is not None and not v.empty}

        if not logs_a or not logs_b:
            st.warning("⚠️ After filtering by date, one of the groups has no games.")
            st.stop()

        # Group summaries
        st.markdown("## 📊 Group Summaries (Per-Game Averages)")
        
        # Calculate PER for each player
        def add_per_to_summary(logs_dict):
            summary = group_summary(logs_dict)
            per_list = []
            for name in summary['Player']:
                if name == 'COMBINED':
                    per_list.append(None)  # Calculate after
                else:
                    per = calculate_per(logs_dict.get(name, pd.DataFrame()))
                    per_list.append(per)
            summary.insert(2, 'PER', per_list)
            return summary
        
        sA = add_per_to_summary(logs_a)
        sB = add_per_to_summary(logs_b)
        
        # Add combined stats row for each group (if more than 1 player)
        def add_combined_row(summary_df):
            if len(summary_df) <= 1:
                return summary_df
            
            # Separate percentage columns from counting stats
            percentage_cols = ["FG_PCT", "FG3_PCT", "FT_PCT"]
            counting_cols = [c for c in summary_df.columns if c not in ["Player", "Games", "PER"] + percentage_cols]
            
            combined_row = {"Player": "COMBINED", "Games": int(summary_df["Games"].sum())}
            
            # For counting stats: sum them
            for col in counting_cols:
                if col in summary_df.columns:
                    combined_row[col] = summary_df[col].sum()
            
            # For percentages: weighted average by games played
            for col in percentage_cols:
                if col in summary_df.columns:
                    # Weighted average: (val1*games1 + val2*games2) / total_games
                    total_games = summary_df["Games"].sum()
                    if total_games > 0:
                        weighted_sum = (summary_df[col] * summary_df["Games"]).sum()
                        combined_row[col] = weighted_sum / total_games
                    else:
                        combined_row[col] = 0
            
            # For PER: weighted average by games played
            if "PER" in summary_df.columns:
                total_games = summary_df["Games"].sum()
                if total_games > 0:
                    # Only use non-null PER values
                    valid_per = summary_df.dropna(subset=['PER'])
                    if not valid_per.empty:
                        weighted_per = (valid_per["PER"] * valid_per["Games"]).sum()
                        combined_row["PER"] = weighted_per / total_games
                    else:
                        combined_row["PER"] = 0
                else:
                    combined_row["PER"] = 0
            
            # Add combined row at the top
            combined_df = pd.DataFrame([combined_row])
            return pd.concat([combined_df, summary_df], ignore_index=True)
        
        sA_with_combined = add_combined_row(sA)
        sB_with_combined = add_combined_row(sB)
        
        ca, cb = st.columns(2)
        with ca:
            st.markdown("### Group A")
            st.dataframe(sA_with_combined, use_container_width=True, hide_index=True)
        with cb:
            st.markdown("### Group B")
            st.dataframe(sB_with_combined, use_container_width=True, hide_index=True)

        # Comparison charts
        st.markdown("## 📈 Group Comparison Over Time")

        metric_options = [
            "PER", "PTS", "REB", "AST", "STL", "BLK", "TOV", 
            "FG_PCT", "FG3_PCT", "FT_PCT", "FG3M", "PLUS_MINUS", "MIN"
        ]
        metric_picks = st.multiselect(
            "Select metrics to visualize",
            metric_options,
            default=["PER", "PTS", "AST", "REB"],
            key="cmp_metrics",
        )
        
        if not metric_picks:
            st.info("💡 Select at least 1 metric to visualize.")
            st.stop()

        view_mode = st.radio(
            "Chart mode",
            ["All metrics on one chart", "One metric at a time"],
            horizontal=True,
            key="cmp_view_mode",
        )

        def group_daily_avg_long(logs: dict, group_name: str, metric: str) -> pd.DataFrame:
            """Create long-format DataFrame for group daily averages."""
            per = group_daily_series(logs, metric)
            if per is None or per.empty:
                return pd.DataFrame(columns=["GAME_DATE", "Group", "Metric", "Value"])
            s = per.mean(axis=1).dropna().sort_index()
            return pd.DataFrame({
                "GAME_DATE": s.index, 
                "Group": group_name, 
                "Metric": metric, 
                "Value": s.values
            })

        # Build data
        parts = []
        for m in metric_picks:
            parts.append(group_daily_avg_long(logs_a, "Group A", m))
            parts.append(group_daily_avg_long(logs_b, "Group B", m))

        long_df = pd.concat(parts, ignore_index=True)
        long_df["GAME_DATE"] = pd.to_datetime(long_df["GAME_DATE"], errors="coerce")
        long_df = long_df.dropna(subset=["GAME_DATE", "Value"])

        if long_df.empty:
            st.info("No data available for the selected metrics in this date window.")
        else:
            if view_mode == "All metrics on one chart":
                long_df["Series"] = long_df["Group"] + " - " + long_df["Metric"]

                line_chart = (
                    alt.Chart(long_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("GAME_DATE:T", title="Game Date"),
                        y=alt.Y("Value:Q", title="Value"),
                        color=alt.Color("Series:N", title="Series"),
                        tooltip=[
                            "Group:N",
                            "Metric:N",
                            alt.Tooltip("GAME_DATE:T", title="Date"),
                            alt.Tooltip("Value:Q", format=".2f"),
                        ],
                    )
                    .properties(height=400)
                    .interactive()
                )
                st.altair_chart(line_chart, use_container_width=True)

            else:
                metric_one = st.selectbox(
                    "Select metric to view", 
                    metric_picks, 
                    index=0, 
                    key="cmp_one_metric"
                )
                df1 = long_df[long_df["Metric"] == metric_one].copy()

                line_chart = (
                    alt.Chart(df1)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("GAME_DATE:T", title="Game Date"),
                        y=alt.Y("Value:Q", title=metric_one),
                        color=alt.Color("Group:N", title="Group"),
                        tooltip=[
                            "Group:N",
                            alt.Tooltip("GAME_DATE:T", title="Date"),
                            alt.Tooltip("Value:Q", format=".2f"),
                        ],
                    )
                    .properties(height=400)
                    .interactive()
                )
                st.altair_chart(line_chart, use_container_width=True)

            st.caption(
                "💡 **Tip:** Avoid mixing percentages (FG%, 3P%, FT%) with counting stats (PTS, REB, AST) "
                "on the same chart. Use 'One metric at a time' for clearer visualization."
            )

        # Delta chart
        st.markdown("## 📊 Group A − Group B (Average Difference)")

        delta_metrics = st.multiselect(
            "Metrics for delta comparison",
            ["PER", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT", "FG3M", "PLUS_MINUS", "MIN"],
            default=["PER", "PTS", "AST", "REB"],
            key="cmp_delta_metrics",
        )

        def mean_metric_from_logs(logs: dict, metric: str) -> float:
            """Calculate mean of a metric across all players in a group."""
            df = group_summary(logs)
            if df is None or df.empty or metric not in df.columns:
                return np.nan
            return float(pd.to_numeric(df[metric], errors="coerce").mean())

        rows = []
        for m in delta_metrics:
            a = mean_metric_from_logs(logs_a, m)
            b = mean_metric_from_logs(logs_b, m)
            if pd.isna(a) or pd.isna(b):
                continue
            rows.append({"Metric": m, "Delta (A-B)": a - b, "Group A": a, "Group B": b})

        delta_df = pd.DataFrame(rows)
        if delta_df.empty:
            st.info("No data available for delta chart.")
        else:
            bar = (
                alt.Chart(delta_df)
                .mark_bar()
                .encode(
                    x=alt.X("Metric:N", sort=None),
                    y=alt.Y("Delta (A-B):Q", title="Delta (A - B)"),
                    color=alt.condition(
                        alt.datum["Delta (A-B)"] > 0,
                        alt.value("steelblue"),
                        alt.value("orange")
                    ),
                    tooltip=[
                        "Metric:N",
                        alt.Tooltip("Group A:Q", format=".2f"),
                        alt.Tooltip("Group B:Q", format=".2f"),
                        alt.Tooltip("Delta (A-B):Q", format=".2f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(bar, use_container_width=True)

        # Synergy matrix (optional)
        if st.checkbox("Show synergy matrix (correlation on overlapping dates)", value=False):
            st.markdown("## 🔗 Synergy Matrix")
            all_logs = {**logs_a, **logs_b}

            # Game count diagnostic
            counts = []
            for n, df in all_logs.items():
                if df is None or df.empty or "GAME_DATE" not in df.columns:
                    continue
                d = pd.to_datetime(df["GAME_DATE"], errors="coerce").dropna().dt.normalize().unique()
                counts.append((n, len(d)))

            diag = pd.DataFrame(counts, columns=["Player", "Games in window"]).sort_values(
                "Games in window", 
                ascending=False
            )
            st.dataframe(diag, use_container_width=True, hide_index=True)
            st.caption(
                "ℹ️ Synergy analysis requires at least 5 overlapping game dates between each pair. "
                "Correlation may be NaN if insufficient overlap."
            )

            metric_for_synergy = metric_picks[0] if metric_picks else "PTS"
            st.write(f"**Correlation metric:** {metric_for_synergy}")
            
            corr = synergy_matrix(all_logs, metric_for_synergy)
            st.dataframe(corr, use_container_width=True)

    # =========================
    # Teams Compare
    # =========================
    with t_teams:
        st.write("Compare 2 teams using recent game statistics.")

        teams_df = load_teams_static()
        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            team_a = st.selectbox("Team A", teams_df["full_name"].tolist(), index=0, key="ta")
        with col2:
            team_b = st.selectbox("Team B", teams_df["full_name"].tolist(), index=1, key="tb")
        with col3:
            last_n = st.slider("Last N games", 5, 25, 10, key="t_last")

        team_a_id = int(teams_df.loc[teams_df["full_name"] == team_a, "id"].iloc[0])
        team_b_id = int(teams_df.loc[teams_df["full_name"] == team_b, "id"].iloc[0])

        # Load logos
        la, lb = None, None
        try:
            la = get_team_logo_url_by_abbr(load_team_abbr(team_a_id))
        except Exception:
            pass
        try:
            lb = get_team_logo_url_by_abbr(load_team_abbr(team_b_id))
        except Exception:
            pass

        # Display team headers
        ha, hb = st.columns(2)
        with ha:
            if la:
                st.image(la, width=TEAM_LOGO_WIDTH)
            st.markdown(f"### {team_a}")
        with hb:
            if lb:
                st.image(lb, width=TEAM_LOGO_WIDTH)
            st.markdown(f"### {team_b}")

        # Load game logs
        try:
            ga = load_team_games(team_a_id, season, last_n)
            gb = load_team_games(team_b_id, season, last_n)
        except Exception as e:
            show_error("Failed to load team game logs. Please try again later.", e)
            st.stop()

        def prep(df: pd.DataFrame, label: str) -> pd.DataFrame:
            """Prepare team data for comparison."""
            out = df.copy()
            if "GAME_DATE" in out.columns:
                out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
                out = out.sort_values("GAME_DATE").set_index("GAME_DATE")
            out.columns = [f"{label} {c}" for c in out.columns]
            return out

        gA = prep(ga, "A")
        gB = prep(gb, "B")
        merged = gA.join(gB, how="outer").sort_index()

        # Display tables
        st.markdown("### 📋 Game Logs")
        cta, ctb = st.columns(2)
        with cta:
            st.dataframe(ga, use_container_width=True, hide_index=True)
        with ctb:
            st.dataframe(gb, use_container_width=True, hide_index=True)

        # Comparison charts
        st.markdown("### 📈 Comparison Charts")
        picks = st.multiselect(
            "Select metrics to compare",
            ["PTS", "FG3M", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV", "STL", "BLK"],
            default=["PTS", "FG3M", "FG_PCT", "REB", "AST", "TOV"],
            key="team_metric_pick",
        )

        for m in picks[:6]:  # Limit to 6 charts
            a_col = f"A {m}"
            b_col = f"B {m}"
            cols = [c for c in [a_col, b_col] if c in merged.columns]
            if cols:
                st.markdown(f"#### {m}")
                st.line_chart(merged[cols], use_container_width=True)
