import re
import pandas as pd

# Paste the full raw fixture text as a triple-quoted string here
raw_fixtures = """
Matchweek 1

Friday 15 August
Liverpool v AFC Bournemouth (Sky Sports)

Saturday 16 August
Aston Villa v Newcastle United (TNT Sports)
Brighton & Hove Albion v Fulham
Nottingham Forest v Brentford
Sunderland v West Ham United
Tottenham Hotspur v Burnley
Wolverhampton Wanderers v Manchester City (Sky Sports)

Sunday 17 August 
Chelsea v Crystal Palace (Sky Sports)
Manchester United v Arsenal (Sky Sports)

Monday 18 August 
Leeds United v Everton (Sky Sports)

MW2 Saturday 23 August 
AFC Bournemouth v Wolverhampton Wanderers
Arsenal v Leeds United
Brentford v Aston Villa
Burnley v Sunderland
Crystal Palace v Nottingham Forest
Everton v Brighton & Hove Albion
Fulham v Manchester United
Manchester City v Tottenham Hotspur
Newcastle United v Liverpool
West Ham United v Chelsea

MW3 Saturday 30 August 
Aston Villa v Crystal Palace
Brighton & Hove Albion v Manchester City
Chelsea v Fulham
Leeds United v Newcastle United
Liverpool v Arsenal
Manchester United v Burnley
Nottingham Forest v West Ham United
Sunderland v Brentford
Tottenham Hotspur v AFC Bournemouth
Wolverhampton Wanderers v Everton

MW4 Saturday 13 September 
AFC Bournemouth v Brighton & Hove Albion
Arsenal v Nottingham Forest
Brentford v Chelsea
Burnley v Liverpool
Crystal Palace v Sunderland
Everton v Aston Villa
Fulham v Leeds United
Manchester City v Manchester United
Newcastle United v Wolverhampton Wanderers
West Ham United v Tottenham Hotspur

MW5 Saturday 20 September
AFC Bournemouth v Newcastle United
Arsenal v Manchester City
Brighton & Hove Albion v Tottenham Hotspur
Burnley v Nottingham Forest
Fulham v Brentford
Liverpool v Everton
Manchester United v Chelsea
Sunderland v Aston Villa
West Ham United v Crystal Palace
Wolverhampton Wanderers v Leeds United

MW6 Saturday 27 September
Aston Villa v Fulham
Brentford v Manchester United
Chelsea v Brighton & Hove Albion
Crystal Palace v Liverpool
Everton v West Ham United
Leeds United v AFC Bournemouth
Manchester City v Burnley
Newcastle United v Arsenal
Nottingham Forest v Sunderland
Tottenham Hotspur v Wolverhampton Wanderers

MW7 Saturday 4 October
AFC Bournemouth v Fulham
Arsenal v West Ham United
Aston Villa v Burnley
Brentford v Manchester City
Chelsea v Liverpool
Everton v Crystal Palace
Leeds United v Tottenham Hotspur
Manchester United v Sunderland
Newcastle United v Nottingham Forest
Wolverhampton Wanderers v Brighton & Hove Albion

MW8 Saturday 18 October
Brighton & Hove Albion v Newcastle United
Burnley v Leeds United
Crystal Palace v AFC Bournemouth
Fulham v Arsenal
Liverpool v Manchester United
Manchester City v Everton
Nottingham Forest v Chelsea
Sunderland v Wolverhampton Wanderers
Tottenham Hotspur v Aston Villa
West Ham United v Brentford

MW9 Saturday 25 October
AFC Bournemouth v Nottingham Forest
Arsenal v Crystal Palace
Aston Villa v Manchester City
Brentford v Liverpool
Chelsea v Sunderland
Everton v Tottenham Hotspur
Leeds United v West Ham United
Manchester United v Brighton & Hove Albion
Newcastle United v Fulham
Wolverhampton Wanderers v Burnley

MW10 Saturday 1 November
Brighton & Hove Albion v Leeds United
Burnley v Arsenal
Crystal Palace v Brentford
Fulham v Wolverhampton Wanderers
Liverpool v Aston Villa
Manchester City v AFC Bournemouth
Nottingham Forest v Manchester United
Sunderland v Everton
Tottenham Hotspur v Chelsea
West Ham United v Newcastle United

MW11 Saturday 8 November
Aston Villa v AFC Bournemouth
Brentford v Newcastle United
Chelsea v Wolverhampton Wanderers
Crystal Palace v Brighton & Hove Albion
Everton v Fulham
Manchester City v Liverpool
Nottingham Forest v Leeds United
Sunderland v Arsenal
Tottenham Hotspur v Manchester United
West Ham United v Burnley

MW12 Saturday 22 November
AFC Bournemouth v West Ham United
Arsenal v Tottenham Hotspur
Brighton & Hove Albion v Brentford
Burnley v Chelsea
Fulham v Sunderland
Leeds United v Aston Villa
Liverpool v Nottingham Forest
Manchester United v Everton
Newcastle United v Manchester City
Wolverhampton Wanderers v Crystal Palace

MW13 Saturday 29 November
Aston Villa v Wolverhampton Wanderers
Brentford v Burnley
Chelsea v Arsenal
Crystal Palace v Manchester United
Everton v Newcastle United
Manchester City v Leeds United
Nottingham Forest v Brighton & Hove Albion
Sunderland v AFC Bournemouth
Tottenham Hotspur v Fulham
West Ham United v Liverpool

MW14 Wednesday 3 December
AFC Bournemouth v Everton
Arsenal v Brentford
Brighton & Hove Albion v Aston Villa
Burnley v Crystal Palace
Fulham v Manchester City
Leeds United v Chelsea
Liverpool v Sunderland
Manchester United v West Ham United
Newcastle United v Tottenham Hotspur
Wolverhampton Wanderers v Nottingham Forest

MW15 Saturday 6 December
AFC Bournemouth v Chelsea
Aston Villa v Arsenal
Brighton & Hove Albion v West Ham United
Everton v Nottingham Forest
Fulham v Crystal Palace
Leeds United v Liverpool
Manchester City v Sunderland
Newcastle United v Burnley
Tottenham Hotspur v Brentford
Wolverhampton Wanderers v Manchester United

MW16 Saturday 13 December
Arsenal v Wolverhampton Wanderers
Brentford v Leeds United
Burnley v Fulham
Chelsea v Everton
Crystal Palace v Manchester City
Liverpool v Brighton & Hove Albion
Manchester United v AFC Bournemouth
Nottingham Forest v Tottenham Hotspur
Sunderland v Newcastle United
West Ham United v Aston Villa

MW17 Saturday 20 December
AFC Bournemouth v Burnley
Aston Villa v Manchester United
Brighton & Hove Albion v Sunderland
Everton v Arsenal
Fulham v Nottingham Forest
Leeds United v Crystal Palace
Manchester City v West Ham United
Newcastle United v Chelsea
Tottenham Hotspur v Liverpool
Wolverhampton Wanderers v Brentford

MW18 Saturday 27 December
Arsenal v Brighton & Hove Albion
Brentford v AFC Bournemouth
Burnley v Everton
Chelsea v Aston Villa
Crystal Palace v Tottenham Hotspur
Liverpool v Wolverhampton Wanderers
Manchester United v Newcastle United
Nottingham Forest v Manchester City
Sunderland v Leeds United
West Ham United v Fulham

MW19 Tuesday 30 December
Arsenal v Aston Villa
Brentford v Tottenham Hotspur
Burnley v Newcastle United
Chelsea v AFC Bournemouth
Crystal Palace v Fulham
Liverpool v Leeds United
Manchester United v Wolverhampton Wanderers
Nottingham Forest v Everton
Sunderland v Manchester City
West Ham United v Brighton & Hove Albion

MW20 Saturday 3 January 2026
AFC Bournemouth v Arsenal
Aston Villa v Nottingham Forest
Brighton & Hove Albion v Burnley
Everton v Brentford
Fulham v Liverpool
Leeds United v Manchester United
Manchester City v Chelsea
Newcastle United v Crystal Palace
Tottenham Hotspur v Sunderland
Wolverhampton Wanderers v West Ham United

MW21 Wednesday 7 January 2026
AFC Bournemouth v Tottenham Hotspur
Arsenal v Liverpool
Brentford v Sunderland
Burnley v Manchester United
Crystal Palace v Aston Villa
Everton v Wolverhampton Wanderers
Fulham v Chelsea
Manchester City v Brighton & Hove Albion
Newcastle United v Leeds United
West Ham United v Nottingham Forest

MW22 Saturday 17 January 2026
Aston Villa v Everton
Brighton & Hove Albion v AFC Bournemouth
Chelsea v Brentford
Leeds United v Fulham
Liverpool v Burnley
Manchester United v Manchester City
Nottingham Forest v Arsenal
Sunderland v Crystal Palace
Tottenham Hotspur v West Ham United
Wolverhampton Wanderers v Newcastle United

MW23 Saturday 24 January 2026
AFC Bournemouth v Liverpool
Arsenal v Manchester United
Brentford v Nottingham Forest
Burnley v Tottenham Hotspur
Crystal Palace v Chelsea
Everton v Leeds United
Fulham v Brighton & Hove Albion
Manchester City v Wolverhampton Wanderers
Newcastle United v Aston Villa
West Ham United v Sunderland

MW24 Saturday 31 January 2026
Aston Villa v Brentford
Brighton & Hove Albion v Everton
Chelsea v West Ham United
Leeds United v Arsenal
Liverpool v Newcastle United
Manchester United v Fulham
Nottingham Forest v Crystal Palace
Sunderland v Burnley
Tottenham Hotspur v Manchester City
Wolverhampton Wanderers v AFC Bournemouth

MW25 Saturday 7 February 2026
AFC Bournemouth v Aston Villa
Arsenal v Sunderland
Brighton & Hove Albion v Crystal Palace
Burnley v West Ham United
Fulham v Everton
Leeds United v Nottingham Forest
Liverpool v Manchester City
Manchester United v Tottenham Hotspur
Newcastle United v Brentford
Wolverhampton Wanderers v Chelsea

MW26 Wednesday 11 February 2026
Aston Villa v Brighton & Hove Albion
Brentford v Arsenal
Chelsea v Leeds United
Crystal Palace v Burnley
Everton v AFC Bournemouth
Manchester City v Fulham
Nottingham Forest v Wolverhampton Wanderers
Sunderland v Liverpool
Tottenham Hotspur v Newcastle United
West Ham United v Manchester United

MW27 Saturday 21 February 2026
Aston Villa v Leeds United
Brentford v Brighton & Hove Albion
Chelsea v Burnley
Crystal Palace v Wolverhampton Wanderers
Everton v Manchester United
Manchester City v Newcastle United
Nottingham Forest v Liverpool
Sunderland v Fulham
Tottenham Hotspur v Arsenal
West Ham United v AFC Bournemouth

MW28 Saturday 28 February 2026
AFC Bournemouth v Sunderland
Arsenal v Chelsea
Brighton & Hove Albion v Nottingham Forest
Burnley v Brentford
Fulham v Tottenham Hotspur
Leeds United v Manchester City
Liverpool v West Ham United
Manchester United v Crystal Palace
Newcastle United v Everton
Wolverhampton Wanderers v Aston Villa

MW29 Wednesday 4 March 2026
AFC Bournemouth v Brentford
Aston Villa v Chelsea
Brighton & Hove Albion v Arsenal
Everton v Burnley
Fulham v West Ham United
Leeds United v Sunderland
Manchester City v Nottingham Forest
Newcastle United v Manchester United
Tottenham Hotspur v Crystal Palace
Wolverhampton Wanderers v Liverpool

MW30 Saturday 14 March 2026
Arsenal v Everton
Brentford v Wolverhampton Wanderers
Burnley v AFC Bournemouth
Chelsea v Newcastle United
Crystal Palace v Leeds United
Liverpool v Tottenham Hotspur
Manchester United v Aston Villa
Nottingham Forest v Fulham
Sunderland v Brighton & Hove Albion
West Ham United v Manchester City

MW31 Saturday 21 March 2026
AFC Bournemouth v Manchester United
Aston Villa v West Ham United
Brighton & Hove Albion v Liverpool
Everton v Chelsea
Fulham v Burnley
Leeds United v Brentford
Manchester City v Crystal Palace
Newcastle United v Sunderland
Tottenham Hotspur v Nottingham Forest
Wolverhampton Wanderers v Arsenal

MW32 Saturday 11 April 2026
Arsenal v AFC Bournemouth
Brentford v Everton
Burnley v Brighton & Hove Albion
Chelsea v Manchester City
Crystal Palace v Newcastle United
Liverpool v Fulham
Manchester United v Leeds United
Nottingham Forest v Aston Villa
Sunderland v Tottenham Hotspur
West Ham United v Wolverhampton Wanderers

MW33 Saturday 18 April 2026
Aston Villa v Sunderland
Brentford v Fulham
Chelsea v Manchester United
Crystal Palace v West Ham United
Everton v Liverpool
Leeds United v Wolverhampton Wanderers
Manchester City v Arsenal
Newcastle United v AFC Bournemouth
Nottingham Forest v Burnley
Tottenham Hotspur v Brighton & Hove Albion

MW34 Saturday 25 April 2026
AFC Bournemouth v Leeds United
Arsenal v Newcastle United
Brighton & Hove Albion v Chelsea
Burnley v Manchester City
Fulham v Aston Villa
Liverpool v Crystal Palace
Manchester United v Brentford
Sunderland v Nottingham Forest
West Ham United v Everton
Wolverhampton Wanderers v Tottenham Hotspur

MW35 Saturday 2 May 2026
AFC Bournemouth v Crystal Palace
Arsenal v Fulham
Aston Villa v Tottenham Hotspur
Brentford v West Ham United
Chelsea v Nottingham Forest
Everton v Manchester City
Leeds United v Burnley
Manchester United v Liverpool
Newcastle United v Brighton & Hove Albion
Wolverhampton Wanderers v Sunderland

MW36 Saturday 9 May 2026
Brighton & Hove Albion v Wolverhampton Wanderers
Burnley v Aston Villa
Crystal Palace v Everton
Fulham v AFC Bournemouth
Liverpool v Chelsea
Manchester City v Brentford
Nottingham Forest v Newcastle United
Sunderland v Manchester United
Tottenham Hotspur v Leeds United
West Ham United v Arsenal

MW37 Sunday 17 May 2026
AFC Bournemouth v Manchester City
Arsenal v Burnley
Aston Villa v Liverpool
Brentford v Crystal Palace
Chelsea v Tottenham Hotspur
Everton v Sunderland
Leeds United v Brighton & Hove Albion
Manchester United v Nottingham Forest
Newcastle United v West Ham United
Wolverhampton Wanderers v Fulham

MW38 Sunday 24 May 2026
Brighton & Hove Albion v Manchester United
Burnley v Wolverhampton Wanderers
Crystal Palace v Arsenal
Fulham v Newcastle United
Liverpool v Brentford
Manchester City v Aston Villa
Nottingham Forest v AFC Bournemouth
Sunderland v Chelsea
Tottenham Hotspur v Everton
West Ham United v Leeds United
"""

# parse into rows
data = []
mw = None
for line in raw_fixtures.splitlines():
    m = re.match(r'Matchweek\s+(\d+)', line)
    if m:
        mw = int(m.group(1))
        continue
    # look for "Team1 v Team2"
    if ' v ' in line:
        # strip off any parenthetical notes
        clean = re.sub(r'\s*\(.*\)$', '', line).strip()
        parts = clean.split(' v ')
        if len(parts) == 2:
            home, away = parts
            data.append({'MatchWeek': mw, 'HomeTeam': home.strip(), 'AwayTeam': away.strip()})

# create DataFrame and save
df_2526 = pd.DataFrame(data)
df_2526.to_csv('./data/fixtures_25_26.csv', index=False)

# display first few rows
