"""
NBA API Client Module

This module provides functions to fetch and process NBA data using the nba_api library.
Includes retry logic, error handling, and data transformation utilities.
"""

from __future__ import annotations

import time
from typing import Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (
    leaguestandings,
    commonteamroster,
    teamgamelog,
    playergamelog,
    teaminfocommon,
)
from nba_api.stats.static import teams as static_teams
from nba_api.stats.static import players as static_players
from nba_api.stats.library.http import STATS_HEADERS

# Import constants
try:
    from constants import (
        API_RETRY_ATTEMPTS,
        API_SLEEP_SECONDS,
        API_TIMEOUT_SECONDS,
        MIN_SYNERGY_GAMES,
    )
except ImportError:
    # Fallback values if constants.py is not available
    API_RETRY_ATTEMPTS = 3
    API_SLEEP_SECONDS = 0.6
    API_TIMEOUT_SECONDS = 30
    MIN_SYNERGY_GAMES = 5


# ----------------------------
# Helper: retry + stability
# ----------------------------
def _call_with_retry(make_call, retries: int = API_RETRY_ATTEMPTS, sleep_sec: float = API_SLEEP_SECONDS):
    """
    Retry API calls with exponential backoff.
    
    Args:
        make_call: Callable that makes the API request
        retries: Number of retry attempts
        sleep_sec: Base sleep duration between retries
        
    Returns:
        Result from the API call
        
    Raises:
        Exception: Last exception if all retries fail
    """
    last_err = None
    for attempt in range(retries):
        try:
            return make_call()
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec * (attempt + 1))
    raise last_err


# ----------------------------
# Standings
# ----------------------------
def get_standings(season: str) -> pd.DataFrame:
    """
    Fetch NBA standings for a given season.
    
    Args:
        season: Season string (e.g., "2024-25")
        
    Returns:
        DataFrame with columns: TeamName, Conference, WINS, LOSSES, WinPCT, Rank
        
    Raises:
        Exception: If API call fails after retries
    """
    def _fetch():
        ep = leaguestandings.LeagueStandings(
            season=season, 
            headers=STATS_HEADERS, 
            timeout=API_TIMEOUT_SECONDS
        )
        return ep.get_data_frames()[0]

    df = _call_with_retry(_fetch)

    possible_rank_cols = ["ConferenceRank", "ConfRank", "PlayoffRank", "Rank"]
    rank_col = next((c for c in possible_rank_cols if c in df.columns), None)

    base_cols = ["TeamName", "Conference", "WINS", "LOSSES", "WinPCT"]
    cols = base_cols + ([rank_col] if rank_col else [])

    out = df[cols].copy()
    if rank_col:
        out.rename(columns={rank_col: "Rank"}, inplace=True)

    if "Rank" not in out.columns:
        out = out.sort_values(["Conference", "WinPCT"], ascending=[True, False])
        out["Rank"] = out.groupby("Conference").cumcount() + 1
    else:
        out = out.sort_values(["Conference", "Rank"])

    for c in ["WINS", "LOSSES", "WinPCT", "Rank"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.reset_index(drop=True)


# ----------------------------
# Teams
# ----------------------------
def get_teams_df() -> pd.DataFrame:
    """
    Get static list of all NBA teams.
    
    Returns:
        DataFrame with team information (id, full_name, abbreviation, etc.)
    """
    all_teams = static_teams.get_teams()
    df = pd.DataFrame(all_teams)
    keep = [c for c in ["id", "full_name", "abbreviation", "nickname", "city"] if c in df.columns]
    out = df[keep].copy()
    if "full_name" in out.columns:
        out = out.sort_values("full_name")
    return out.reset_index(drop=True)


def get_team_id_by_name(team_name: str) -> int:
    """
    Find team ID by name (exact or partial match).
    
    Args:
        team_name: Team name or nickname
        
    Returns:
        Team ID
        
    Raises:
        ValueError: If team not found
    """
    all_teams = static_teams.get_teams()

    match = next((t for t in all_teams if t.get("nickname") == team_name), None)
    if match:
        return int(match["id"])

    match = next((t for t in all_teams if team_name.lower() in t.get("full_name", "").lower()), None)
    if not match:
        raise ValueError(f"Could not find team id for: {team_name}")
    return int(match["id"])


def get_team_abbr(team_id: int) -> str:
    """
    Get team abbreviation by team ID.
    
    Args:
        team_id: NBA team ID
        
    Returns:
        Team abbreviation in lowercase (e.g., "lal" for Lakers)
        
    Raises:
        ValueError: If team info cannot be retrieved
    """
    def _fetch():
        ep = teaminfocommon.TeamInfoCommon(
            team_id=team_id, 
            headers=STATS_HEADERS, 
            timeout=API_TIMEOUT_SECONDS
        )
        return ep.get_data_frames()[0]

    df = _call_with_retry(_fetch)
    if df.empty:
        raise ValueError("TeamInfoCommon returned empty dataframe")

    col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else None
    if not col:
        raise ValueError("TEAM_ABBREVIATION not found in TeamInfoCommon result")

    return str(df.loc[0, col]).strip().lower()


def get_team_logo_url_by_abbr(team_abbr_lower: str) -> str:
    """
    Get team logo URL from abbreviation.
    
    Args:
        team_abbr_lower: Team abbreviation in lowercase
        
    Returns:
        URL to team logo SVG
    """
    return f"https://raw.githubusercontent.com/gtkacz/nba-logo-api/main/icons/{team_abbr_lower}.svg"


def get_team_roster(team_id: int, season: str) -> pd.DataFrame:
    """
    Get team roster for a given season.
    
    Args:
        team_id: NBA team ID
        season: Season string (e.g., "2024-25")
        
    Returns:
        DataFrame with roster information
    """
    def _fetch():
        ep = commonteamroster.CommonTeamRoster(
            team_id=team_id, 
            season=season, 
            headers=STATS_HEADERS, 
            timeout=API_TIMEOUT_SECONDS
        )
        return ep.get_data_frames()[0]

    df = _call_with_retry(_fetch)

    keep = [
        c for c in ["PLAYER", "PLAYER_ID", "NUM", "POSITION", "HEIGHT", "WEIGHT", "AGE", "EXP", "SCHOOL"]
        if c in df.columns
    ]
    out = df[keep].copy()
    if "PLAYER_ID" in out.columns:
        out["PLAYER_ID"] = pd.to_numeric(out["PLAYER_ID"], errors="coerce")
    return out.reset_index(drop=True)


def get_team_roster_player_ids(team_id: int, season: str) -> pd.DataFrame:
    """
    Get simplified roster with just player names and IDs.
    
    Args:
        team_id: NBA team ID
        season: Season string
        
    Returns:
        DataFrame with PLAYER, PLAYER_ID, POSITION columns
    """
    roster = get_team_roster(team_id, season)
    keep = [c for c in ["PLAYER", "PLAYER_ID", "POSITION"] if c in roster.columns]
    out = roster[keep].copy().dropna()
    out["PLAYER_ID"] = pd.to_numeric(out["PLAYER_ID"], errors="coerce")
    return out.dropna().reset_index(drop=True)


def get_team_last_games(team_id: int, season: str, last_n: Optional[int] = 10) -> pd.DataFrame:
    """
    Get team game log for a given season.
    
    Args:
        team_id: NBA team ID
        season: Season string (e.g., "2024-25")
        last_n: Number of most recent games to return. If None, returns full season.
        
    Returns:
        DataFrame with game log data sorted by date (most recent first if last_n is set)
        
    Raises:
        Exception: If API call fails
    """
    def _fetch():
        ep = teamgamelog.TeamGameLog(
            team_id=team_id, 
            season=season, 
            headers=STATS_HEADERS, 
            timeout=API_TIMEOUT_SECONDS
        )
        return ep.get_data_frames()[0]

    df = _call_with_retry(_fetch)

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE", ascending=False)

    keep = [c for c in [
        "GAME_DATE", "MATCHUP", "WL",
        "PTS", "REB", "AST", "TOV",
        "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "STL", "BLK",
    ] if c in df.columns]

    out = df[keep].copy()

    if last_n is not None:
        out = out.head(int(last_n))

    return out.reset_index(drop=True)


# ----------------------------
# Players
# ----------------------------
def search_players(query: str, limit: int = 20) -> pd.DataFrame:
    """
    Search for players by name.
    
    Args:
        query: Player name search query
        limit: Maximum number of results
        
    Returns:
        DataFrame with player search results
    """
    results = static_players.find_players_by_full_name(query)
    df = pd.DataFrame(results)
    if df.empty:
        return df
    keep = [c for c in ["full_name", "id", "is_active"] if c in df.columns]
    df = df[keep].copy()
    if "is_active" in df.columns:
        df = df.sort_values("is_active", ascending=False)
    return df.head(limit).reset_index(drop=True)


def get_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    """
    Get player game log for a given season.
    
    Args:
        player_id: NBA player ID
        season: Season string (e.g., "2024-25")
        
    Returns:
        DataFrame with game-by-game statistics
        
    Raises:
        Exception: If API call fails
    """
    def _fetch():
        ep = playergamelog.PlayerGameLog(
            player_id=player_id, 
            season=season, 
            headers=STATS_HEADERS, 
            timeout=API_TIMEOUT_SECONDS
        )
        return ep.get_data_frames()[0]

    df = _call_with_retry(_fetch)

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE", ascending=True)

    keep = [c for c in [
        "GAME_DATE", "MATCHUP", "WL",
        "MIN",
        "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "PLUS_MINUS",
    ] if c in df.columns]

    out = df[keep].copy().reset_index(drop=True)
    return out


def _to_num(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric type.
    
    Args:
        df: Input DataFrame
        cols: Column names to convert
        
    Returns:
        DataFrame with converted columns
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def summarize_player(df_log: pd.DataFrame) -> pd.Series:
    """
    Calculate per-game averages for a player.
    
    Args:
        df_log: Player game log DataFrame
        
    Returns:
        Series with averaged statistics and game count
    """
    if df_log.empty:
        return pd.Series(dtype="float64")

    df = add_derived_player_columns(df_log)

    numeric_cols = [c for c in [
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
        "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "PLUS_MINUS",
    ] if c in df.columns]

    out = df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)
    out["Games"] = len(df)
    return out


def summarize_players(df_log: pd.DataFrame, player_name: str) -> pd.Series:
    """
    Summarize player statistics with player name included.
    
    Args:
        df_log: Player game log DataFrame
        player_name: Player's name
        
    Returns:
        Series with player name, games, and averaged statistics
    """
    if df_log.empty:
        return pd.Series({"Player": player_name, "Games": 0})

    s = summarize_player(df_log)
    out = pd.Series({"Player": player_name, "Games": int(len(df_log))})
    for k, v in s.items():
        if k not in ["Games"]:
            out[k] = v
    return out


# ----------------------------
# Bias splits
# ----------------------------
def _is_home_game(matchup: str) -> bool:
    """
    Determine if game was played at home based on matchup string.
    
    Args:
        matchup: Matchup string (e.g., "LAL vs. GSW" or "LAL @ GSW")
        
    Returns:
        True if home game, False if away
    """
    return " vs " in str(matchup) or " vs. " in str(matchup)


def compute_player_splits(df_log: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate player statistics split by home/away and win/loss.
    
    Args:
        df_log: Player game log DataFrame
        
    Returns:
        DataFrame with splits (Home, Away, Win, Loss) and their averages
    """
    if df_log.empty:
        return pd.DataFrame()

    df = add_derived_player_columns(df_log)

    if "MATCHUP" in df.columns:
        df["is_home"] = df["MATCHUP"].apply(_is_home_game)
    else:
        df["is_home"] = np.nan

    if "WL" in df.columns:
        df["is_win"] = df["WL"].astype(str).eq("W")
    else:
        df["is_win"] = np.nan

    numeric = [c for c in [
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
        "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "PLUS_MINUS",
    ] if c in df.columns]

    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def summarize(mask, label):
        sub = df.loc[mask, numeric]
        row = sub.mean(numeric_only=True)
        row["Games"] = int(mask.sum())
        row["Split"] = label
        return row

    rows = []
    if df["is_home"].notna().any():
        rows.append(summarize(df["is_home"] == True, "Home"))
        rows.append(summarize(df["is_home"] == False, "Away"))
    if df["is_win"].notna().any():
        rows.append(summarize(df["is_win"] == True, "Win"))
        rows.append(summarize(df["is_win"] == False, "Loss"))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    cols = ["Split", "Games"] + [c for c in numeric if c in out.columns]
    return out[cols].reset_index(drop=True)


def recent_vs_season(df_log: pd.DataFrame, last_n: int = 10) -> pd.DataFrame:
    """
    Compare recent performance vs season average.
    
    Args:
        df_log: Player game log DataFrame
        last_n: Number of recent games to compare
        
    Returns:
        DataFrame comparing season average, recent average, and difference
    """
    if df_log.empty:
        return pd.DataFrame()

    df = add_derived_player_columns(df_log)
    if "GAME_DATE" in df.columns:
        df = df.sort_values("GAME_DATE")

    metrics = [c for c in [
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "FG3M",
        "FGA", "FTA",
        "PLUS_MINUS",
    ] if c in df.columns]

    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    season_avg = df[metrics].mean(numeric_only=True)
    recent_avg = df.tail(last_n)[metrics].mean(numeric_only=True)

    out = pd.DataFrame({"Season avg": season_avg, f"Last {last_n} avg": recent_avg})
    out["Diff"] = out[f"Last {last_n} avg"] - out["Season avg"]
    return out.reset_index(names="Metric")


def add_rolling_metrics(
    df_log: pd.DataFrame, 
    metrics: Iterable[str], 
    windows: Tuple[int, ...] = (5, 10)
) -> pd.DataFrame:
    """
    Add rolling average columns for specified metrics.
    
    Args:
        df_log: Player game log DataFrame
        metrics: List of metric column names
        windows: Tuple of window sizes for rolling averages
        
    Returns:
        DataFrame with added rolling average columns (e.g., PTS_roll5, PTS_roll10)
    """
    if df_log.empty or "GAME_DATE" not in df_log.columns:
        return df_log

    df = add_derived_player_columns(df_log).copy().sort_values("GAME_DATE")
    for m in metrics:
        if m not in df.columns:
            continue
        df[m] = pd.to_numeric(df[m], errors="coerce")
        for w in windows:
            df[f"{m}_roll{w}"] = df[m].rolling(w, min_periods=max(2, w // 2)).mean()
    return df


# ----------------------------
# Synergy / Group compare
# ----------------------------
def align_logs_on_dates(
    log_a: pd.DataFrame, 
    log_b: pd.DataFrame, 
    metric: str = "PTS"
) -> Tuple[pd.Series, pd.Series]:
    """
    Align two player logs on common game dates for correlation analysis.
    
    Args:
        log_a: First player's game log
        log_b: Second player's game log
        metric: Metric to align (default: "PTS")
        
    Returns:
        Tuple of (series_a, series_b) with values on common dates
    """
    if log_a.empty or log_b.empty or "GAME_DATE" not in log_a.columns or "GAME_DATE" not in log_b.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if metric not in log_a.columns or metric not in log_b.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    a = log_a[["GAME_DATE", metric]].copy()
    b = log_b[["GAME_DATE", metric]].copy()

    a["GAME_DATE"] = pd.to_datetime(a["GAME_DATE"], errors="coerce")
    b["GAME_DATE"] = pd.to_datetime(b["GAME_DATE"], errors="coerce")
    a[metric] = pd.to_numeric(a[metric], errors="coerce")
    b[metric] = pd.to_numeric(b[metric], errors="coerce")

    merged = a.merge(b, on="GAME_DATE", how="inner", suffixes=("_a", "_b")).dropna()
    return merged[f"{metric}_a"], merged[f"{metric}_b"]


def synergy_matrix(player_logs: Dict[str, pd.DataFrame], metric: str = "PTS") -> pd.DataFrame:
    """
    Calculate correlation matrix between multiple players on a given metric.
    
    Args:
        player_logs: Dictionary mapping player names to their game logs
        metric: Metric to calculate correlations for
        
    Returns:
        DataFrame with correlation matrix (player x player)
    """
    names = list(player_logs.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                mat.loc[ni, nj] = 1.0
            elif pd.notna(mat.loc[ni, nj]):
                continue
            else:
                s1, s2 = align_logs_on_dates(player_logs[ni], player_logs[nj], metric=metric)
                if len(s1) >= MIN_SYNERGY_GAMES and len(s2) >= MIN_SYNERGY_GAMES:
                    corr = float(pd.Series(s1).corr(pd.Series(s2)))
                else:
                    corr = np.nan
                mat.loc[ni, nj] = corr
                mat.loc[nj, ni] = corr

    return mat


def group_summary(player_logs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create summary statistics for a group of players.
    
    Args:
        player_logs: Dictionary mapping player names to their game logs
        
    Returns:
        DataFrame with per-game averages for each player
    """
    rows = []
    for name, log in player_logs.items():
        rows.append(summarize_players(log, name))
    df = pd.DataFrame(rows)

    order = [
        "Player", "Games",
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "FG3M",
        "FGA", "FTA",
        "PLUS_MINUS",
    ]
    cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    
    if "PTS" in df.columns:
        df = df.sort_values(["PTS"], ascending=False, na_position="last")
    return df[cols].reset_index(drop=True)


def group_daily_series(player_logs: Dict[str, pd.DataFrame], metric: str = "PTS") -> pd.DataFrame:
    """
    Create a pivot table of daily metric values for multiple players.
    
    Args:
        player_logs: Dictionary mapping player names to their game logs
        metric: Metric to track (default: "PTS")
        
    Returns:
        DataFrame with dates as index and players as columns
    """
    rows = []
    for name, log in player_logs.items():
        if log.empty or "GAME_DATE" not in log.columns or metric not in log.columns:
            continue
        df = log[["GAME_DATE", metric]].copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df["Player"] = name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    big = pd.concat(rows, ignore_index=True).dropna()
    pivot = big.pivot_table(index="GAME_DATE", columns="Player", values=metric, aggfunc="mean").sort_index()
    return pivot


def calculate_per(df_log: pd.DataFrame) -> float:
    """
    Calculate Player Efficiency Rating (PER) - John Hollinger's formula.
    Simplified version for per-game stats.
    
    Args:
        df_log: Player game log DataFrame
        
    Returns:
        PER score (league average is 15.0)
    """
    if df_log.empty:
        return 0.0
    
    df = add_derived_player_columns(df_log)
    
    # Get averages
    stats = df.apply(pd.to_numeric, errors='coerce').mean()
    
    # Extract stats (use .get() to handle missing columns)
    pts = stats.get('PTS', 0)
    fgm = stats.get('FGM', 0)
    fga = stats.get('FGA', 0)
    ftm = stats.get('FTM', 0)
    fta = stats.get('FTA', 0)
    fg3m = stats.get('FG3M', 0)
    reb = stats.get('REB', 0)
    ast = stats.get('AST', 0)
    stl = stats.get('STL', 0)
    blk = stats.get('BLK', 0)
    tov = stats.get('TOV', 0)
    pf = stats.get('PF', 0) if 'PF' in stats else 2.0  # Default fouls
    
    # Simplified PER formula (per-game version)
    # Positive contributions
    per = (
        pts * 1.0 +
        reb * 0.4 +
        ast * 0.7 +
        stl * 1.0 +
        blk * 1.0 +
        fg3m * 0.5 +
        fgm * 0.5 -
        # Negative contributions
        (fga - fgm) * 0.5 -  # Missed field goals
        (fta - ftm) * 0.5 -  # Missed free throws
        tov * 1.0 -
        pf * 0.5
    )
    
    # Scale to league average of ~15
    per = max(0, per * 0.67)  # Scaling factor
    
    return round(per, 1)


def add_derived_player_columns(df_log: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived/convenience columns for downstream analysis.
    Safe to call even if columns are missing.
    
    Args:
        df_log: Player game log DataFrame
        
    Returns:
        DataFrame with numeric conversions and derived columns added
    """
    if df_log is None or df_log.empty:
        return df_log

    df = df_log.copy()

    # Convert likely numeric columns
    num_cols = [
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
        "PLUS_MINUS",
        "FG_PCT", "FG3_PCT", "FT_PCT",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def filter_log_by_date(
    df_log: pd.DataFrame, 
    start_date=None, 
    end_date=None
) -> pd.DataFrame:
    """
    Filter game log to date range [start_date, end_date] inclusive.
    
    Args:
        df_log: Game log DataFrame
        start_date: Start date (datetime.date, datetime, or None)
        end_date: End date (datetime.date, datetime, or None)
        
    Returns:
        Filtered DataFrame
    """
    if df_log is None or df_log.empty or "GAME_DATE" not in df_log.columns:
        return df_log

    df = df_log.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    if start_date is not None:
        start = pd.to_datetime(start_date)
        df = df[df["GAME_DATE"] >= start]

    if end_date is not None:
        end = pd.to_datetime(end_date)
        df = df[df["GAME_DATE"] <= end]

    return df.reset_index(drop=True)


def common_date_window(player_logs: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Find the common date range across all player logs.
    
    Args:
        player_logs: Dictionary mapping player names to their game logs
        
    Returns:
        Tuple of (min_date, max_date) or (None, None) if no dates found
    """
    mins = []
    maxs = []
    for _, df in player_logs.items():
        if df is None or df.empty or "GAME_DATE" not in df.columns:
            continue
        d = pd.to_datetime(df["GAME_DATE"], errors="coerce").dropna()
        if d.empty:
            continue
        mins.append(d.min())
        maxs.append(d.max())
    if not mins:
        return None, None
    return min(mins), max(maxs)
