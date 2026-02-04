

# API Configuration
API_RETRY_ATTEMPTS = 3
API_SLEEP_SECONDS = 0.6  # Increased from 0.35 for better rate limit handling
API_TIMEOUT_SECONDS = 30

# Cache Configuration
CACHE_TTL_HOURS = 6

# UI Configuration
HEADSHOT_WIDTH = 95
TEAM_LOGO_WIDTH = 70
HEADSHOT_URL_TEMPLATE = "https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
TEAM_LOGO_URL_TEMPLATE = "https://raw.githubusercontent.com/gtkacz/nba-logo-api/main/icons/{team_abbr}.svg"

# Available Seasons
SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23"]

# Metrics Configuration
PLAYER_BASIC_METRICS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
PLAYER_SHOOTING_METRICS = ["FG_PCT", "FG3_PCT", "FT_PCT", "FG3M"]
PLAYER_ADVANCED_METRICS = ["MIN", "PLUS_MINUS"]

TEAM_METRICS = [
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
    "FG3M", "REB", "AST", "TOV", "STL", "BLK"
]

# Default Values
DEFAULT_LAST_N_GAMES = 20
DEFAULT_ROLLING_WINDOWS = (5, 10)
MIN_SYNERGY_GAMES = 5
