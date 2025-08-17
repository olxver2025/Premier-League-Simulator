import pandas as pd
from collections import defaultdict

# 1) Define a factory for a fresh stats dict
def stats_factory():
    return {'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'P': 0, 'Pts': 0}

# 2) Create a nested defaultdict: season → team → stats dict
season_team_stats = defaultdict(lambda: defaultdict(stats_factory))

# 3) Load your full dataset (with header row)
df = pd.read_csv("data/PremierLeague.csv")

# 4) Keep only the columns we need
df = df[['Season', 'HomeTeam', 'AwayTeam', 'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals']]

# 5) Iterate rows and update stats
for _, row in df.iterrows():
    season = row['Season']
    home = row['HomeTeam']
    away = row['AwayTeam']
    # Skip rows where goals are missing or non-numeric
    try:
        hg = int(row['FullTimeHomeTeamGoals'])
        ag = int(row['FullTimeAwayTeamGoals'])
    except:
        continue

    # Home team
    hs = season_team_stats[season][home]
    hs['GF'] += hg
    hs['GA'] += ag
    hs['P']  += 1
    if hg > ag:
        hs['W']  += 1
        hs['Pts'] += 3
    elif hg == ag:
        hs['D']  += 1
        hs['Pts'] += 1
    else:
        hs['L']  += 1

    # Away team
    as_ = season_team_stats[season][away]
    as_['GF'] += ag
    as_['GA'] += hg
    as_['P']  += 1
    if ag > hg:
        as_['W']  += 1
        as_['Pts'] += 3
    elif ag == hg:
        as_['D']  += 1
        as_['Pts'] += 1
    else:
        as_['L']  += 1

# 6) Convert to a flat DataFrame
rows = []
for season, teams in season_team_stats.items():
    for team, s in teams.items():
        rows.append({
            'Season': season,
            'Team': team,
            'W': s['W'],
            'D': s['D'],
            'L': s['L'],
            'GF': s['GF'],
            'GA': s['GA'],
            'GD': s['GF'] - s['GA'],
            'Pts': s['Pts']
        })

standings_df = pd.DataFrame(rows)
standings_df = standings_df.sort_values(
    by=['Season', 'Pts', 'GD', 'GF'],
    ascending=[True, False, False, False]
)

# 7) Save to CSV
standings_df.to_csv("data/season_table.csv", index=False)
print("✅ Created data/season_table.csv")
