import streamlit as st
import pandas as pd
import numpy as np
import torch
import scipy.stats as stats
from scipy.optimize import minimize
import pickle
import os
import re
from difflib import get_close_matches
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("CUDA available?:", torch.cuda.is_available())
print("CUDA toolkit version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
# ─── 1) PAGE CONFIG & CSV PATHS ────────────────────────────────────────────────
st.set_page_config(page_title="🏟️ Advanced PL Simulator", layout="wide")
MATCH_DATA_CSV      = "data/PremierLeague.csv"
FIXTURES_2425_CSV   = "data/2024-2025 fixtures.csv"
FIXTURES_2526_CSV   = "data/fixtures_25_26.csv"
PARAMS_PICKLE       = "data/dixon_coles_params.pkl"

# ─── 2) LOAD ALL DATA ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    md   = pd.read_csv(MATCH_DATA_CSV, parse_dates=["Date"], dayfirst=True)
    fx25 = pd.read_csv(FIXTURES_2425_CSV)
    fx26 = pd.read_csv(FIXTURES_2526_CSV)
    return md, fx25, fx26

match_df, fixtures_df_2425, fixtures_df_2526 = load_data()
for df in (fixtures_df_2425, fixtures_df_2526):
    df.columns = df.columns.str.strip()

# ─── 3) DYNAMIC COLUMN DETECTION FOR MATCH DATA ────────────────────────────────
match_df.columns = match_df.columns.str.strip()
home_col = next(c for c in match_df.columns if "HomeTeam" in c)
away_col = next(c for c in match_df.columns if "AwayTeam" in c)
fthg_col = next(c for c in match_df.columns if "FullTimeHomeTeamGoals" in c)
ftag_col = next(c for c in match_df.columns if "FullTimeAwayTeamGoals" in c)






# ─── 4) DETECT HOME/AWAY IN FIXTURES ───────────────────────────────────────────
def detect_cols(df):
    cols = df.columns.tolist()
    home = next((c for c in cols if "home" in c.lower()), cols[0])
    away = next((c for c in cols if "away" in c.lower()), cols[1])
    return home, away

fx25_home, fx25_away = detect_cols(fixtures_df_2425)
fx26_home, fx26_away = detect_cols(fixtures_df_2526)

# ─── 5) UNION TEAM LIST & INDEX ────────────────────────────────────────────────
teams = sorted(
    set(fixtures_df_2425[fx25_home]) | set(fixtures_df_2425[fx25_away]) |
    set(fixtures_df_2526[fx26_home]) | set(fixtures_df_2526[fx26_away])
)
team_idx = {t: i for i, t in enumerate(teams)}
n_teams = len(teams)

# ─── 6) FILTER HISTORICAL MATCHES FOR CURRENT TEAMS ────────────────────────────
match_df = match_df[
    match_df[home_col].isin(teams) &
    match_df[away_col].isin(teams)
]

# ─── 7) TRAINING DATA FILTER ──────────────────────────────────────────────────
recent_seasons = sorted(match_df["Season"].unique())[-5:]
train_df = match_df[match_df["Season"].isin(recent_seasons)]

# ─── 8) VECTORISED DIXON–COLES FIT (UNCHANGED) ─────────────────────────────────
@st.cache_data
def fit_dixon_coles(df):
    hc = next(c for c in df.columns if "HomeTeam" in c)
    ac = next(c for c in df.columns if "AwayTeam" in c)
    gh = next(c for c in df.columns if "FullTimeHomeTeamGoals" in c)
    ga = next(c for c in df.columns if "FullTimeAwayTeamGoals" in c)
    idx_h = df[hc].map(team_idx).to_numpy()
    idx_a = df[ac].map(team_idx).to_numpy()
    g_h   = df[gh].to_numpy()
    g_a   = df[ga].to_numpy()
    μh, μa = g_h.mean(), g_a.mean()
    x0 = np.concatenate([np.ones(4*n_teams), [0.0]])

    def neg_ll(p):
        α = p[0:n_teams]
        β = p[n_teams:2*n_teams]
        γ = p[2*n_teams:3*n_teams]
        δ = p[3*n_teams:4*n_teams]
        ρ = p[-1]
        λ = α[idx_h] * δ[idx_a] * μh
        μ = β[idx_a] * γ[idx_h] * μa
        
        ll = stats.poisson.logpmf(g_h, λ).sum() + stats.poisson.logpmf(g_a, μ).sum()
        mask = (g_h <= 1) & (g_a <= 1)
        if mask.any():
            ll += np.log(1 - ρ * np.exp(-λ[mask]) * (1-λ[mask]) *
                         np.exp(-μ[mask]) * (1-μ[mask])).sum()
        return -ll

    with st.spinner("🔧 Fitting Dixon–Coles…"):
        res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter":100})

    α, β, γ, δ = np.split(res.x[:-1], [n_teams, 2*n_teams, 3*n_teams])
    ρ = res.x[-1]
    return α, β, γ, δ, ρ, μh, μa

# load or fit & cache
if os.path.exists(PARAMS_PICKLE):
    α, β, γ, δ, rho, μh, μa, teams_c, idx_c = pickle.load(open(PARAMS_PICKLE, "rb"))
    if teams_c != teams:
        α, β, γ, δ, rho, μh, μa = fit_dixon_coles(train_df)
        pickle.dump((α, β, γ, δ, rho, μh, μa, teams, team_idx), open(PARAMS_PICKLE, "wb"))
else:
    α, β, γ, δ, rho, μh, μa = fit_dixon_coles(train_df)
    pickle.dump((α, β, γ, δ, rho, μh, μa, teams, team_idx), open(PARAMS_PICKLE, "wb"))


α_t  = torch.tensor(α,  dtype=torch.float32, device=device)
β_t  = torch.tensor(β,  dtype=torch.float32, device=device)
γ_t  = torch.tensor(γ,  dtype=torch.float32, device=device)
δ_t  = torch.tensor(δ,  dtype=torch.float32, device=device)
μh_t = torch.tensor(μh, dtype=torch.float32, device=device)
μa_t = torch.tensor(μa, dtype=torch.float32, device=device)

# ─── NOW define prepare_torch (it will see α_t, δ_t, etc. as globals) ─────────
def prepare_torch(fixtures):
    h_idx = torch.tensor([team_idx[h] for h,a in fixtures],
                         dtype=torch.int64, device=device)
    a_idx = torch.tensor([team_idx[a] for h,a in fixtures],
                         dtype=torch.int64, device=device)
    lam = α_t[h_idx] * δ_t[a_idx] * μh_t
    mu  = β_t[a_idx] * γ_t[h_idx] * μa_t
    return h_idx, a_idx, lam, mu


def simulate_batch(
    fixtures: list[tuple[str,str]],
    promoted: list[str] = None,
    bias_factor: float   = 0.5,
    n_sims: int          = 1024
):
    """Returns dict of tensors (n_sims × n_teams) for P,W,D,L,GF,GA,Pts."""
    promoted   = promoted or []
    n_teams    = len(teams)

    # build promo mask once
    promo_mask = torch.zeros(n_teams, device=device)
    promo_mask[[team_idx[t] for t in promoted]] = 1.0

    # precompute fixture arrays
    h_idx, a_idx, base_lam, base_mu = prepare_torch(fixtures)

    M = h_idx.numel()                          # number of matches
    stat_shape  = (n_sims, n_teams)              # for P,W,D,L,GF,GA tensors
    match_shape = (n_sims, M)                   # for hg/ag & index expansion

    # 1) sample all goals in match-space
    lam = base_lam.unsqueeze(0).expand(match_shape)
    mu  = base_mu.unsqueeze(0).expand(match_shape)
    lam = lam * (bias_factor * promo_mask[h_idx].unsqueeze(0)
                 + (1 - promo_mask[h_idx]).unsqueeze(0))
    mu  = mu  * (bias_factor * promo_mask[a_idx].unsqueeze(0)
                 + (1 - promo_mask[a_idx]).unsqueeze(0))

    hg = torch.poisson(lam).to(torch.int32)      # shape (n_sims, M)
    ag = torch.poisson(mu).to(torch.int32)       # shape (n_sims, M)

    # 2) init stats in team-space
    GF  = torch.zeros(stat_shape, dtype=torch.int32, device=device)
    GA  = torch.zeros_like(GF)
    W   = torch.zeros_like(GF)
    D   = torch.zeros_like(GF)
    L   = torch.zeros_like(GF)

    # 3) expand indices into match-space
    h_idx_e = h_idx.unsqueeze(0).expand(match_shape)  
    a_idx_e = a_idx.unsqueeze(0).expand(match_shape)  

    # 4) scatter goals
    GF.scatter_add_(1, h_idx_e, hg)
    GA.scatter_add_(1, h_idx_e, ag)
    GF.scatter_add_(1, a_idx_e, ag)
    GA.scatter_add_(1, a_idx_e, hg)

    # 5) compute result masks
    home_win = (hg > ag).to(torch.int32)
    away_win = (ag > hg).to(torch.int32)
    draw     = (hg == ag).to(torch.int32)

    # 6) scatter W/D/L
    W.scatter_add_(1, h_idx_e, home_win)
    W.scatter_add_(1, a_idx_e, away_win)
    L.scatter_add_(1, h_idx_e, away_win)
    L.scatter_add_(1, a_idx_e, home_win)
    D.scatter_add_(1, h_idx_e, draw)
    D.scatter_add_(1, a_idx_e, draw)

    # 7) points & played
    Pts = 3*W + D
    P   = W + D + L

    return {"P":P, "W":W, "D":D, "L":L, "GF":GF, "GA":GA, "Pts":Pts}

def simulate_until_vectorized(
    cond_str: str,
    fixtures: list[tuple[str,str]],
    promoted: list[str]    = None,
    bias_factor: float      = 0.5,
    max_sims: int           = 7500
):
    promoted = promoted or []
    # ─── identify this season's 20 clubs & their indices in the union ─────────
    season_teams = sorted({h for h,_ in fixtures} | {a for _,a in fixtures})
    season_idx   = [team_idx[t] for t in season_teams]
    nseason      = len(season_teams)

    # ─── parse conditions with the correct team-count ───────────────────────────
    descs = parse_conditions(cond_str, nseason)

    # ─── run one big GPU batch for ALL union teams ─────────────────────────────
    stats_union = simulate_batch(
        fixtures,
        promoted=promoted,
        bias_factor=bias_factor,
        n_sims=max_sims
    )
    # subset to just our 20 clubs
    P  = stats_union["Pts"][:, season_idx]       # (max_sims, 20)
    GF = stats_union["GF"][:,  season_idx]
    GA = stats_union["GA"][:,  season_idx]
    GD = GF - GA

    W  = stats_union["W"][:, season_idx]
    D  = stats_union["D"][:, season_idx]
    L  = stats_union["L"][:, season_idx]

    # ─── build sort key & compute ranks in one shot ────────────────────────────
    key = P.to(torch.int64)*10_000 + GD.to(torch.int64)*100 + GF.to(torch.int64)
    sorted_idx = torch.argsort(key, dim=1, descending=True)  # (max_sims,20)
    ranks = torch.empty_like(sorted_idx)
    ranks.scatter_(
        1, sorted_idx,
        torch.arange(nseason, device=device)
             .unsqueeze(0).expand_as(sorted_idx)
    )  # ranks[i,j] is 0-based finish of season_teams[j] in sim i

    # ─── build a mask of sims meeting *all* your conditions ───────────────────
    mask = torch.ones(max_sims, dtype=torch.bool, device=device)
    for kind, team, val in descs:
        tidx = season_teams.index(team)
        if kind == "finish":
            mask &= (ranks[:, tidx] == (val-1))
        elif kind == "finish_ge":
            mask &= (ranks[:, tidx] >= (val-1))
        elif kind == "unbeaten":
            mask &= (stats_union["L"][:, team_idx[team]] == 0)
        elif kind == "gf":
            mask &= (GF[:, tidx] >= val)
        elif kind == "ga_lt":
            mask &= (GA[:, tidx] < val)
        elif kind == "pts":
            mask &= (P[:, tidx] >= val)
        elif kind == "gd":
            mask &= (GD[:, tidx] >= val)
        elif kind == "w_eq":
            mask &= (W[:,   tidx] == val)
        elif kind == "d_eq":
            mask &= (D[:,   tidx] == val)
        elif kind == "l_eq":
            mask &= (D[:,   tidx] == val)

    hits = torch.nonzero(mask, as_tuple=False)
    if hits.numel() == 0:
        return None, max_sims

    i = hits[0].item()   # first matching sim

    # ─── reconstruct that one table on CPU ────────────────────────────────────
    arrs = {k: v[i, season_idx].cpu().numpy() for k,v in stats_union.items()}
    df = pd.DataFrame({
        "Team": season_teams,
        "P":    arrs["P"],
        "W":    arrs["W"],
        "D":    arrs["D"],
        "L":    arrs["L"],
        "GF":   arrs["GF"],
        "GA":   arrs["GA"],
        "Pts":  arrs["Pts"],
    })
    df["GD"] = df["GF"] - df["GA"]
    df = df.sort_values(["Pts","GD","GF"], ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Pos"

    return df, i+1

def simulate_until_batch(
    cond_str: str,
    fixtures: list[tuple[str,str]],
    promoted: list[str]    = None,
    bias_factor: float      = 0.5,
    max_sims: int           = 7500
):
    """One‐shot GPU: simulate max_sims seasons, then scan for your condition."""
    promoted     = promoted or []
    conds        = parse_conditions(cond_str)
    # only these 20 clubs this season
    season_teams = sorted({h for h,_ in fixtures} | {a for _,a in fixtures})

    # 1) fire off a single big batch on the GPU
    stats = simulate_batch(
        fixtures,
        promoted=promoted,
        bias_factor=bias_factor,
        n_sims=max_sims
    )

    # 2) pull everything to CPU as numpy arrays
    arrs = {k: v.cpu().numpy() for k,v in stats.items()}

    # 3) scan in Python but all data is local—no more GPU launches
    for i in range(max_sims):
        # build DataFrame for sim #i
        df = pd.DataFrame({
            "Team": teams,
            "P":    arrs["P"][i],
            "W":    arrs["W"][i],
            "D":    arrs["D"][i],
            "L":    arrs["L"][i],
            "GF":   arrs["GF"][i],
            "GA":   arrs["GA"][i],
            "Pts":  arrs["Pts"][i],
        })
        df["GD"] = df["GF"] - df["GA"]

        # filter to just this season’s 20 clubs & re-rank
        df = df[df.Team.isin(season_teams)]
        df = (
            df
            .sort_values(["Pts","GD","GF"], ascending=False)
            .reset_index(drop=True)
        )
        df.index = df.index + 1
        df.index.name = "Pos"

        # check your conditions
        if all(cond(df) for cond in conds):
            return df, i+1

    return None, max_sims
# ─── 9) SIMULATION FUNCTIONS WITH PROMOTION BIAS ───────────────────────────────
def simulate_season(fixtures, promoted: list[str]=None, bias_factor: float=0.5):
    """GPU‐accelerated single‐season sim."""
    promoted    = promoted or []
    n_teams     = len(teams)
    # boolean mask on GPU
    promo_mask  = torch.zeros(n_teams, device=device)
    promo_mask[[team_idx[t] for t in promoted]] = 1.0

    # get index tensors + base lambdas
    h_idx, a_idx, base_lam, base_mu = prepare_torch(fixtures)

    # apply bias
    lam = base_lam * (bias_factor * promo_mask[h_idx] + (1 - promo_mask[h_idx]))
    mu  = base_mu  * (bias_factor * promo_mask[a_idx] + (1 - promo_mask[a_idx]))

    # draw all goals in parallel
    hg = torch.poisson(lam).to(torch.int32)
    ag = torch.poisson(mu).to(torch.int32)

    # prepare stat tensors as int32
    GF  = torch.zeros(n_teams, dtype=torch.int32, device=device)
    GA  = torch.zeros(n_teams, dtype=torch.int32, device=device)
    W   = torch.zeros(n_teams, dtype=torch.int32, device=device)
    D   = torch.zeros(n_teams, dtype=torch.int32, device=device)
    L   = torch.zeros(n_teams, dtype=torch.int32, device=device)
    Pts = torch.zeros(n_teams, dtype=torch.int32, device=device)

    # accumulate goals
    GF.index_add_(0, h_idx, hg)
    GA.index_add_(0, h_idx, ag)
    GF.index_add_(0, a_idx, ag)
    GA.index_add_(0, a_idx, hg)

    # wins/draws/losses
    home_win = (hg > ag).to(torch.int32)
    away_win = (ag > hg).to(torch.int32)
    draw     = (hg == ag).to(torch.int32)

    W.index_add_(0, h_idx, home_win)
    W.index_add_(0, a_idx, away_win)
    L.index_add_(0, h_idx, away_win)
    L.index_add_(0, a_idx, home_win)
    D.index_add_(0, h_idx, draw)
    D.index_add_(0, a_idx, draw)

    Pts.index_add_(0, h_idx, 3*home_win + draw)
    Pts.index_add_(0, a_idx, 3*away_win + draw)

    # matches played = W+D+L
    P = W + D + L

    # bring back to CPU and build DataFrame
    df = pd.DataFrame({
        'Team': teams,
        'P':   P.cpu().numpy(),
        'W':   W.cpu().numpy(),
        'D':   D.cpu().numpy(),
        'L':   L.cpu().numpy(),
        'GF':  GF.cpu().numpy(),
        'GA':  GA.cpu().numpy(),
        'Pts': Pts.cpu().numpy(),
    })
    df['GD'] = df['GF'] - df['GA']

    return (
        df
        .sort_values(['Pts','GD','GF'], ascending=False)
        .reset_index(drop=True)
        .reset_index().rename(columns={'index':'Pos'})
        .set_index('Pos')
    )



def simulate_season_custom(
    ratings: dict[str, float],
    fixtures_order,
    promoted: list[str] = None,
    bias_factor: float = 0.5
):
    # unused atm
    """Star‐rating sim with same promoted‐team bias."""
    promoted = promoted or []
    season_teams = sorted({h for h, _ in fixtures_order} |
                          {a for _, a in fixtures_order})
    μ_avg = (μh + μa) / 2.0
    mults = {
        t: 0.5 + 0.2 * ratings.get(t, 2.5)
        for t in season_teams
    }
    stats = {
        t: {'P':0, 'W':0, 'D':0, 'L':0, 'GF':0, 'GA':0, 'Pts':0}
        for t in season_teams
    }

    for h, a in fixtures_order:
        lam = μ_avg * mults[h] * (bias_factor if h in promoted else 1.0)
        mu  = μ_avg * (2 - mults[a]) * (bias_factor if a in promoted else 1.0)

        hg, ag = np.random.poisson(lam), np.random.poisson(mu)
        sh, sa = stats[h], stats[a]

        sh['P']  += 1; sh['GF'] += hg; sh['GA'] += ag
        sa['P']  += 1; sa['GF'] += ag; sa['GA'] += hg

        if hg > ag:
            sh['W']  += 1; sh['Pts'] += 3
            sa['L']  += 1
        elif hg < ag:
            sa['W']  += 1; sa['Pts'] += 3
            sh['L']  += 1
        else:
            sh['D']  += 1; sh['Pts'] += 1
            sa['D']  += 1; sa['Pts'] += 1

    df = pd.DataFrame(stats).T
    df['GD'] = df['GF'] - df['GA']
    df = (
        df.sort_values(['Pts','GD','GF'], ascending=False)
          .reset_index().rename(columns={'index':'Team'})
    )
    df.index      = range(1, len(df)+1)
    df.index.name = 'Pos'
    return df




ALIAS_MAP = {
    # Manchester United
    "man utd":            "Manchester United",
    "man u":              "Manchester United",
    "manchester united":  "Manchester United",

    # Manchester City
    "man city":           "Manchester City",
    "manchester city":    "Manchester City",

    # Liverpool
    "lfc":                "Liverpool",
    "liverpool fc":       "Liverpool",
    "liverpool":          "Liverpool",

    # Tottenham
    "spurs":              "Tottenham Hotspur",
    "tottenham":          "Tottenham Hotspur",
    "tottenham hotspur":  "Tottenham Hotspur",

    "newcastle":          "Newcastle United",
    "newcastle united":   "Newcastle United",

    "forest":             "Nottingham Forest",
    "nottingham":         "Nottingham Forest",

    "bournemouth":        "AFC Bournemouth",
    "bournemouth afc":    "AFC Bournemouth",
    "brighton":           "Brighton & Hove Albion",
    "brighton & hove":    "Brighton & Hove Albion",

    "villa":             "Aston Villa",

    "leeds":              "Leeds United",

    "palace":            "Crystal Palace",

    "west ham":         "West Ham United",

    "wolves":             "Wolverhampton Wanderers",
    "wolverhampton":      "Wolverhampton Wanderers",
    
}

def canonical_team(name: str) -> str:
    key = name.lower().strip()
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    for t in teams:
        if t.lower() == key:
            return t
    # fuzzy fallback
    matches = get_close_matches(key, [t.lower() for t in teams], n=1, cutoff=0.6)
    if matches:
        return next(t for t in teams if t.lower() == matches[0])
    raise ValueError(f"Unknown team: '{name}'")


# ─── 7) CUSTOM CONDITION PARSER ───────────────────────────────────────────────
def parse_conditions(cond_str: str, n_teams: int):
    """
    Break cond_str into a list of (kind, team_name, value) descriptors.
    kind is one of:
      - 'finish'      exact finish position
      - 'finish_ge'   finish ≥ value (for relegation/bottom)
      - 'unbeaten'
      - 'gf'          goals for ≥ value
      - 'ga_lt'       goals against < value
      - 'pts'         points ≥ value
      - 'gd'          goal difference ≥ value
      - 'w'           wins ≥ value
      - 'd'           draws ≥ value
    """
    descs = []
    clauses = re.split(r'\s+and\s+', cond_str, flags=re.IGNORECASE)
    for cl in clauses:
        text = cl.lower()

        # (1) resolve team via aliases/fuzzy match…
        team = None
        for alias, real in ALIAS_MAP.items():
            if alias in text:
                team = real
                break
        if not team:
            for t in teams:
                if t.lower() in text:
                    team = t
                    break
        if not team:
            continue

        # 1) “finishes Nth”
        m = re.search(r'finish(?:es)?\s+(\d+)', text)
        if m:
            descs.append(("finish", team, int(m.group(1))))
            continue

        # 2) “wins league” ⇒ finish 1
        if re.search(r'\bwin(?:s)?\b.*\bleague\b', text):
            descs.append(("finish", team, 1))
            continue

        # 3) relegated / bottom ⇒ finish ≥ n_teams - 2
        if 'relegat' in text or 'bottom' in text:
            descs.append(("finish_ge", team, n_teams - 2))
            continue

        # 4) unbeaten
        if re.search(r'(invincible|unbeaten)', text):
            descs.append(("unbeaten", team, None))
            continue

        # 5) “N+ goals”
        m = re.search(r'(\d+)\s*\+?\s*(?:goals|scored)', text)
        if m:
            descs.append(("gf", team, int(m.group(1))))
            continue

        # 6) “conceded fewer than N”
        m = re.search(r'conceded.*?(\d+)', text)
        if m:
            descs.append(("ga_lt", team, int(m.group(1))))
            continue

        # 7) “N+ points”
        m = re.search(r'(\d+)\s*\+?\s*points', text)
        if m:
            descs.append(("pts", team, int(m.group(1))))
            continue

        # 8) “goal difference ≥ N”
        m = re.search(r'goal difference.*?(\d+)', text)
        if m:
            descs.append(("gd", team, int(m.group(1))))
            continue

        # 9) “wins N games” (or “win N games” / “won N games”)
        m = re.search(r'(?:win|wins|won)\s+(\d+)(?:\s+games?)?', text)
        if m:
            descs.append(("w_eq", team, int(m.group(1))))
            continue

        # 10) draws N games (exact)
        m = re.search(r'(?:draw|draws|drew)\s+(\d+)(?:\s+games?)?', text)
        if m:
            descs.append(("d_eq", team, int(m.group(1))))
            continue
        # 11) loses N games (exact)
        m = re.search(r'(?:loses|lost|lose)\s+(\d+)(?:\s+games?)?', text)
        if m:
            descs.append(("l_eq", team, int(m.group(1))))
            continue

    if not descs:
        raise ValueError(f"No valid conditions found in: '{cond_str}'")
    return descs



def simulate_until(
    cond_str: str,
    fixtures_order: list[tuple[str,str]],
    promoted: list[str]    = None,
    bias_factor: float      = 0.5,
    max_tries: int          = 5000
):
    """
    Simulate only the `fixtures_order` you pass in (so 24/25 or 25/26),
    apply `promoted` bias, and stop once ANY custom condition is met.
    """
    conds    = parse_conditions(cond_str)
    promoted = promoted or []

    for i in range(1, max_tries + 1):
        if i % 100 == 0:
            st.sidebar.info(f"🔄 Tried {i} sims…")
        # this will only build a table for the 20 teams in your fixtures_order
        tbl = simulate_season(fixtures_order, promoted=promoted, bias_factor=bias_factor)
        if all(c(tbl) for c in conds):
            st.sidebar.success(f"✅ Met at simulation #{i}")
            return tbl, i

    st.sidebar.warning(f"⚠️ Not met after {max_tries} sims")
    return None, max_tries