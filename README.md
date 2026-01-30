# 🏀 NBA Analytics Dashboard

An interactive web application for comprehensive NBA statistics analysis, built with **Streamlit** and **nba_api**.


## 🌐 Live Demo

**Try it now:** [NBA Analytics Dashboard](https://nba-players-and-teams-analytics-dashboard.streamlit.app/)


## ✨ Features

### 📊 **Standings Dashboard**
- View current NBA standings for Eastern and Western conferences
- Filter by conference, minimum wins, and team search
- Sort by rank, win percentage, and conference
- Real-time season data (2022-23 through 2025-26)

### 🏀 **Team Analysis**
**Single Team Mode:**
- Complete team roster with player details
- Last N games performance tracking
- Interactive trend charts for key metrics (PTS, 3PM, FG%, etc.)
- Customizable game range (5-82 games)

**Team Comparison Mode:**
- Compare multiple teams side-by-side
- Date range or "Last N games" filtering
- Multi-metric trend visualization (connected line charts)
- Average performance bar charts
- Support for comparing 2+ teams simultaneously

### 👤 **Player Explorer**
**Player Search Options:**
- Global player search across all NBA teams
- Team roster-based player selection
- Player headshot and team logo display


### 🔄 **Compare & Synergy**
**Player Group Comparison:**
- Build custom player pools via search or roster
- Divide players into Group A and Group B
- Date range filtering with quick presets (Last 7/14/30 games)
- Per-game averages for each group
- Interactive trend charts with dual-group visualization
- Delta analysis (Group A - Group B)
- Correlation matrix for synergy analysis

**Team Comparison:**
- Head-to-head comparison of 2 teams
- Last N games analysis (5-25 games)
- Side-by-side game logs
- Multi-metric comparison charts
- Visual team logos and branding




## 🚀 Installation

### Prerequisites
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Poetry**: For dependency management ([Install Poetry](https://python-poetry.org/docs/#installation))




### Using the Dashboard

#### **Standings Page**
1. Select conference (All/East/West)
2. Set minimum wins filter
3. Search for specific teams
4. View results in tabbed interface

#### **Team Analysis**
1. Choose "Single Team" or "Compare Teams"
2. Select team(s) from dropdown
3. Adjust game range with slider
4. Explore roster and game logs
5. Analyze trend charts

#### **Player Explorer**
1. Find player via global search or team roster
2. View season statistics and averages
3. Explore tabs: Overview, Splits, Trends, Game Log
4. Analyze rolling averages and shooting volume

#### **Compare & Synergy**
1. Add players to pool (via search or roster)
2. Organize into Group A and Group B
3. Select date range or use quick presets
4. Compare averages and trends
5. Analyze delta charts and correlations

---

## 📁 Project Structure

```
nba-analytics-dashboard/
│
├── main.py                 # Main Streamlit application (EXECUTABLE FILE)
├── nba_client.py          # NBA API client with data fetching functions
├── constants.py           # Configuration constants and settings
│
├── pyproject.toml         # Poetry dependencies and project metadata
├── poetry.lock            # Poetry lock file (DO NOT EDIT MANUALLY)
├── README.md              # This file
│
├── .gitignore             # Git ignore rules
│
└── docs/                  # Documentation (optional)
    └── screenshots/       # Application screenshots
```



## 🛠️ Tech Stack

### Core Technologies
- **[Streamlit](https://streamlit.io/)** (1.x) - Web application framework
- **[nba_api](https://github.com/swar/nba_api)** - NBA Stats API wrapper
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Altair](https://altair-viz.github.io/)** - Declarative visualization

### Development Tools
- **[Poetry](https://python-poetry.org/)** - Dependency management
- **Python 3.10+** - Programming language

### APIs Used
- **NBA Stats API** - Official NBA statistics endpoint
- **nba_api** - Python wrapper for stats.nba.com

---

## ⚙️ Configuration

### Customizing Settings

Edit `constants.py` to customize:

## 🐛 Troubleshooting

Enable debug mode in the sidebar (⚙️ Settings) to see:
- Detailed error messages
- API response information
- Stack traces


