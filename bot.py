import json
import math
import datetime
import sqlite3
import sys
import pandas as pd
import numpy as np
import torch
import os, platform, time, psutil
import nextcord, cooldowns
from nextcord import Interaction, SlashOption
from nextcord.ext import commands, application_checks

from main import canonical_team, α_t, β_t, γ_t, δ_t, μh_t, μa_t, team_idx
from typing import Optional, Callable, Awaitable

__all__ = ["QuickButton", "style_from_name"]

match_df = pd.read_csv("data/PremierLeague.csv", parse_dates=["Date"], dayfirst=True)
match_df.columns = match_df.columns.str.strip()

home_col = next(c for c in match_df.columns if "HomeTeam" in c)
away_col = next(c for c in match_df.columns if "AwayTeam" in c)
fthg_col = next(c for c in match_df.columns if "FullTimeHomeTeamGoals" in c)
ftag_col = next(c for c in match_df.columns if "FullTimeAwayTeamGoals" in c)
def style_from_name(name: str) -> nextcord.ButtonStyle:
    """Translate a human‑friendly **colour name** to :class:`nextcord.ButtonStyle`."""
    mapping = {
        "primary": nextcord.ButtonStyle.primary,
        "secondary": nextcord.ButtonStyle.secondary,
        "success": nextcord.ButtonStyle.success,
        "danger": nextcord.ButtonStyle.danger,
        "link": nextcord.ButtonStyle.link,
    }
    return mapping.get(name.lower(), nextcord.ButtonStyle.primary)


class QuickButton(nextcord.ui.View):
    """A `View` containing a **single** button that can be edited onto any message.

    Parameters
    ----------
    label: str
        Text displayed on the button.
    emoji: str | nextcord.PartialEmoji | None
        Emoji to show next to the label.
    url: str | None
        If supplied the button becomes a *link button* (no interaction payload).
    color: str, default ``"primary"``
        One of ``primary``, ``secondary``, ``success``, ``danger``, or ``link``.
        Ignored when *url* is given.
    custom_id: str | None
        Custom identifier used for non‑link buttons.
    disabled: bool, default ``False``
        Create the button in a disabled state.
    timeout: float, default ``180``
        Seconds before Discord stops listening for interactions.
    callback: Coroutine | None
        Custom coroutine executed when the button is pressed (non‑link only).
    """

    def __init__(
        self,
        *,
        label: str,
        emoji: Optional[str] = None,
        url: Optional[str] = None,
        color: str = "primary",
        custom_id: Optional[str] = None,
        disabled: bool = False,
        timeout: float = 180,
        callback: Optional[Callable[[nextcord.Interaction], Awaitable[None]]] = None,
    ) -> None:
        super().__init__(timeout=timeout)

        style = nextcord.ButtonStyle.link if url else style_from_name(color)

        self.button: nextcord.ui.Button = nextcord.ui.Button(
            label=label,
            emoji=emoji,
            style=style,
            url=url,
            custom_id=custom_id,
            disabled=disabled,
        )
        self.add_item(self.button)

        # Hook up interaction callback for non‑link buttons.
        if style is not nextcord.ButtonStyle.link:
            if callback is None:

                async def _default(inter: nextcord.Interaction):
                    await inter.response.send_message(
                        f"You clicked **{label}**!", ephemeral=True
                    )

                self.button.callback = _default
            else:
                self.button.callback = callback

    async def attach_to(self, message: nextcord.Message) -> None:
        """Edit *message* to display this view in place."""
        await message.edit(view=self)


# ─── 1) LOAD BOTH SEASONS’ FIXTURES ──────────────────────────────────────────
# 24/25 has a "Date" column
# ─── 1) LOAD FIXTURES & DETECT HOME/AWAY/WEEK ───────────────────────────────
fx_2425 = pd.read_csv("data/2024-2025 fixtures.csv")
fx_2526 = pd.read_csv("data/epl-2025-GMTStandardTime.csv", parse_dates=["Date"], dayfirst=True)
fx_2526.columns = fx_2526.columns.str.strip()

# now detect “home”, “away” and “round” columns automatically:

def detect_cols_full(df):
    cols = df.columns.tolist()
    home = next(c for c in cols if "home"  in c.lower())
    away = next(c for c in cols if "away"  in c.lower())
    week = next(c for c in cols if "round" in c.lower()
                        or "week"  in c.lower()
                        or "matchweek" in c.lower())
    return home, away, week

h25, a25, w25 = detect_cols_full(fx_2425)
h26, a26, w26 = detect_cols_full(fx_2526)

# ─── 2) BUILD FIXTURE LISTS FOR SIMULATION ──────────────────────────────────
fixture_list_2425 = list(zip(fx_2425[h25], fx_2425[a25]))
fixture_list_2526 = list(zip(fx_2526[h26], fx_2526[a26]))

fx_2526[h26] = fx_2526[h26].apply(canonical_team)
fx_2526[a26] = fx_2526[a26].apply(canonical_team)

# now build your zipped fixture list
fixture_list_2526 = list(zip(fx_2526[h26], fx_2526[a26]))

# ─── 4) PROMOTION BIAS CONFIG ────────────────────────────────────────────────
PROMOTED = {
    "24/25": ["Ipswich", "Leicester", "Southampton"],
    "25/26": ["Leeds United", "Sunderland", "Burnley"]
}
BIAS_FACTOR = 0.35

# ─── 5) IMPORT YOUR CORE SIMULATION LOGIC ─────────────────────────────────────
from main import (
    simulate_batch,
    simulate_season,
    simulate_until,
    simulate_until_batch,
    simulate_until_vectorized,
    teams, team_idx,
    α, β, γ, δ, μh, μa
)

intents = nextcord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot( intents=intents)
START_TIME = datetime.datetime.now(datetime.timezone.utc)

# updates:
bot.updates = [
        {
            "version": "1.1a",
            "changes": [
                "❓ New /faq command to see questions and answers for topics",
                "🌱 New seed logic implementation; see /faq seeds",
                "💾 Save simulation button under /simulate custom-condition",
                "👁️ View past simulations that you've saved with /view simulations",
                "🦾 Minor increase to simulation speed and performance",
                "🐛 Fixed minor bugs",
                "⚠️ The bot may slow down during peak times due to high load, please be patient",
                "⚠️ This update may take longer to roll out"
            ]
        },
        {
            "version": "1.0a",
            "changes": [
                "💫 Bot moved from beta to stable release",
                "🚀 Major increase to simulation speed and performance (100,000 simulations in under a second)",
                "⬆️ Increased max simulation attempts to 100,000 for /simulate custom-condition",
                "🆕 Added more conditions for /simulate custom-condition",
                "🐛 Fixed minor bugs",
                "✅ Re-enabled /simulate gameweek for 25/26 season",
                "⚠️ Removed 24/25 season from all /simulate commands" ,
            ]
        },
        {
            "version": "0.3b",
            "changes": [
                "🐛 Fixed major issues relating to new 25/26 season implementation",
                "⬆️ Increased simulation attempts to 7500 for /simulate custom-condition",
                "⚠️ Disabled /simulate gameweek temporarily due to issues relating to new fixtures",

            ]
        },
        {
            "version": "0.3",
            "changes": [
                "🆕 Added 2025/2026 season and real-life fixtures to /simulate commands",
                "🐛 Fixed general issues and bugs",
                "🦾 Improved performance and response times",
                "⚠️ Disabled /simulate gameweek temporarily due to issues",
                "⚠️ /simulate custom-condition now has a max of 5000 simulation attempts",
                "⚠️ Some commands now have cooldowns to prevent abuse",
                "⚠️ Some bugs may be present, please report them using /bug"
            ]
        },
        {
            "version": "0.2",
            "changes": [
                "📈 Added /simulate custom-condition to simulate until a condition is met"
            ]
        },
        {
            "version": "0.1",
            "changes": [
                "🚀 Initial launch of bot with basic features and slow simulation speed",
            ]
        }
    ]


# ─── 6) PAGINATOR ────────────────────────────────────────────────────────────
class PaginatorView(nextcord.ui.View):
    def __init__(self, embeds):
        super().__init__(timeout=None)
        self.embeds = embeds
        self.current = 0

    @nextcord.ui.button(label="⬅️ Prev", style=nextcord.ButtonStyle.secondary)
    async def prev(self, button, inter: Interaction):
        self.current = (self.current - 1) % len(self.embeds)
        await inter.response.edit_message(embed=self.embeds[self.current], view=self)

    @nextcord.ui.button(label="Next ➡️", style=nextcord.ButtonStyle.secondary)
    async def nxt(self, button, inter: Interaction):
        self.current = (self.current + 1) % len(self.embeds)
        await inter.response.edit_message(embed=self.embeds[self.current], view=self)

# ─── 7) BOT EVENTS ──────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print(f"Bot ready as {bot.user}")
    # create a table under data/users.db called settings
    conn = sqlite3.connect("data/users.db")
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS settings (
            user_id TEXT PRIMARY KEY,
            beta_mode BOOLEAN DEFAULT FALSE,
            auto_save BOOLEAN DEFAULT FALSE

        )
        '''
    )
    conn.commit()
    conn.close()




@bot.event
async def on_application_command_error(inter, error):
    error = getattr(error, "original", error)
    if isinstance(error, cooldowns.CallableOnCooldown):
        return await inter.response.send_message(
            f"⏳ You’re rate-limited! Try again in {error.retry_after:.0f}s", 
            ephemeral=True
        )
    raise error

# ─── 8) PARENT COMMAND ───────────────────────────────────────────────────────
@bot.slash_command(name="simulate")
async def simulate_parent(inter: Interaction):
    pass

# ─── 9) /simulate table ─────────────────────────────────────────────────────
@simulate_parent.subcommand(
    name="table",
    description="Simulate a full PL season"
)
@cooldowns.cooldown(5, 30, bucket=cooldowns.SlashBucket.author)
async def simulate_table(
    interaction: Interaction,
    season: str = SlashOption(
        name="season",
        description="Pick season",
        choices={"2025/26": "25/26"},
        required=True
    )
):
    original_message = await interaction.response.send_message(
        "<a:simulating:1383927364349726862> Simulating, this may take a while...",
        ephemeral=True
    )
    fixtures = fixture_list_2425 if season == "24/25" else fixture_list_2526
    promoted = PROMOTED[season]

    # run the sim across the full union of teams
    full_table = simulate_season(fixtures, promoted=promoted, bias_factor=BIAS_FACTOR)

    # keep only the 20 clubs actually in this season
    season_teams = {h for h, _ in fixtures} | {a for _, a in fixtures}
    table = full_table[full_table["Team"].isin(season_teams)]

    # re-index positions 1…20
    table = (
        table
        .reset_index(drop=True)           # drop old Pos
        .reset_index()                    # bring back old integer index
        .rename(columns={"index": "Pos"})
        .set_index("Pos")
    )

    pages = math.ceil(len(table) / 10)
    embeds = []
    for p in range(pages):
        chunk = table.iloc[p*10:(p+1)*10]
        e = nextcord.Embed(
            title=f"📊 Simulated Table {season}",
            description="Based on model predictions",
            color=0x3498DB
        )
        desc = "\n".join(
            f"**{pos}. {r.Team}** — {r.Pts} pts "
            f"({r.W}W {r.D}D {r.L}L, {r.GF}GF {r.GA}GA {r.GD}GD)"
            for pos, r in chunk.iterrows()
        )
        e.add_field(name="Standings", value=desc, inline=False)
        e.set_footer(text=f"Page {p+1}/{pages}")
        embeds.append(e)

    await original_message.edit("", embed=embeds[0], view=PaginatorView(embeds))

# ─── 10) /simulate custom-condition ──────────────────────────────────────────
from seed_generation import (
    DB_PATH,
    fetch_user_sims,
    make_seed,
    simulations_json_lookup,
    log_global_sim,
)
import random
class SaveablePaginatorView(PaginatorView):
    def __init__(self, embeds, user, season, condition, seed, result_df, *, auto_save: bool = False):
        super().__init__(embeds)
        self.user = user
        self.season = season
        self.condition = condition
        self.seed = seed
        self.result_json = result_df.to_json(orient='records')

        # create Save button
        self.save_button = nextcord.ui.Button(
            label="💾 Save Simulation",
            style=nextcord.ButtonStyle.success,
            custom_id="save_simulation"
        )
        # disable (and relabel) if auto-save already happened
        if auto_save:
            self.save_button.disabled = True
            self.save_button.label = "💾 Auto-saved"
            self.save_button.style = nextcord.ButtonStyle.secondary

        self.add_item(self.save_button)
        self.save_button.callback = self.save_callback

    async def save_callback(self, interaction: Interaction):
        # disable button to prevent duplicates
        self.save_button.disabled = True
        await interaction.response.edit_message(view=self)

        # persist
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        full_cond = f"{self.season}|{self.condition}"
        cur.execute(
            "INSERT INTO simulations (user_id, condition, seed, result_json) VALUES (?, ?, ?, ?)",
            (str(self.user.id), full_cond, self.seed, self.result_json)
        )
        conn.commit()
        conn.close()
        await interaction.response.send_message("✅ Simulation saved!", ephemeral=True)





@simulate_parent.subcommand(
    name="custom-condition",
    description="Simulate until a condition is met"
)
@cooldowns.cooldown(2, 15, bucket=cooldowns.SlashBucket.author)
async def simulate_condition(
    interaction: Interaction,
    season: str = SlashOption(
        name="season",
        description="Pick season",
        choices={"2025/26": "25/26"},
        required=True
    ),
    condition: str = SlashOption(
        name="condition",
        description="e.g. Man City win league and Sunderland relegated",
        required=True
    ),
    seed: str = SlashOption(
        name="seed",
        description="Optional seed for reproducible sims",
        required=False
    )
):
    # initial “working…” response
    original = await interaction.response.send_message(
        "<a:simulating:1383927364349726862> Simulating, this may take a while...",
        ephemeral=True
    )

    # pick correct fixture list & promos
    fixtures = fixture_list_2425 if season == "24/25" else fixture_list_2526
    promoted = PROMOTED[season]

    # ─── Seed logic ─────────────────────────────────────────────────────────
    if seed:
        chosen_seed = seed
    else:
        chosen_seed = make_seed(None)
    seed_int = int.from_bytes(chosen_seed.encode('utf-8'), 'little') % (2**32)
    torch.manual_seed(seed_int)
    np.random.seed(seed_int)
    random.seed(seed_int)
    # ────────────────────────────────────────────────────────────────────────

    # run it
    try:
        t0 = time.perf_counter()
        tbl, tries = simulate_until_vectorized(
            condition, fixtures,
            promoted=promoted,
            bias_factor=BIAS_FACTOR,
            max_sims=100000
        )
        sim_secs = time.perf_counter() - t0

        # record run for global statistics
        full_cond = f"{season}|{condition}"
        log_global_sim(str(interaction.user.id), full_cond)

        if tbl is None:
            return await original.edit("❌ Couldn't meet that condition in 100,000 sims.")
    except ValueError:
        return await original.edit(f"❌ Invalid condition: `{condition}`")
    except Exception as e:
        print("Simulation error:", e)
        return await original.edit("❌ Unexpected error; please try again later.")

    conn_u = sqlite3.connect(USER_DB_PATH)
    cur_u = conn_u.cursor()
    cur_u.execute(
        "SELECT auto_save FROM settings WHERE user_id = ?",
        (str(interaction.user.id),)
    )
    row = cur_u.fetchone()
    auto_save_enabled = bool(row[0]) if row else False
    conn_u.close()

    # if auto-save is on, persist immediately (and let the user know)
    if auto_save_enabled:
        conn_s = sqlite3.connect(DB_PATH)
        cur_s = conn_s.cursor()
        full_cond = f"{season}|{condition}"
        cur_s.execute(
            "INSERT INTO simulations (user_id, condition, seed, result_json) VALUES (?, ?, ?, ?)",
            (str(interaction.user.id), full_cond, chosen_seed, tbl.to_json(orient='records'))
        )
        conn_s.commit()
        conn_s.close()
    # build embeds and pass auto_save flag into the view
    pages = math.ceil(len(tbl) / 10)
    embeds = []
    for p in range(pages):
        chunk = tbl.iloc[p*10:(p+1)*10]
        e = nextcord.Embed(
            title=f"🔍 Custom Condition {season}",
            description=(
                f"Condition: `{condition}`\n"
                f"Simulations needed: **{tries}**  Chance ≈ **{100/tries:.4f}%**\n"
                f"Time took: **{sim_secs:.2f}s**"
            ),
            color=0x2ECC71
        )
        desc = "\n".join(
            f"**{pos}. {r.Team}** — {r.Pts} pts "
            f"({r.W}W {r.D}D {r.L}L, {r.GF}GF {r.GA}GA {r.GD}GD)"
            for pos, r in chunk.iterrows()
        )
        e.add_field(name="Standings", value=desc, inline=False)
        e.set_footer(text=f"Seed: {chosen_seed} • Page {p+1}/{pages}")
        embeds.append(e)

    view = SaveablePaginatorView(
        embeds=embeds,
        user=interaction.user,
        season=season,
        condition=condition,
        seed=chosen_seed,
        result_df=tbl,
        auto_save=auto_save_enabled
    )
    await original.edit(content=None, embed=embeds[0], view=view)


# ─── /view simulations ──────────────────────────────────────────────────────

@bot.slash_command(name="view", description="View parent command")
async def view(interaction: Interaction):
    pass


@view.subcommand(name="simulations", description="View your saved simulations")
async def view_simulations(inter: Interaction):
    rows = fetch_user_sims(str(inter.user.id), limit=10)
    if not rows:
        return await inter.response.send_message(
            "You have no saved simulations.", ephemeral=True
        )

    # build first embed
    sim_id, cond, seed, created = rows[0]
    embed = nextcord.Embed(
        title="📂 Your Saved Simulations",
        description=(
            f"**ID {sim_id}**\n"
            f"Condition: `{cond}`\n"
            f"Seed: `{seed}`\n"
            f"Saved: {created}"
        ),
        color=0x3498DB
    )
    embed.set_footer(text=f"Page 1/{len(rows)}")

    view = SavedSimulationsView(rows, inter.user)
    await inter.response.send_message(embed=embed, view=view, ephemeral=True)


class SavedSimulationsView(nextcord.ui.View):
    def __init__(self, rows: list, user: nextcord.User, *, timeout=None):
        super().__init__(timeout=timeout)
        self.rows = rows
        self.user = user
        self.current = 0

    async def _update_embed(self, interaction: Interaction):
        sim_id, cond, seed, created = self.rows[self.current]
        embed = nextcord.Embed(
            title="📂 Your Saved Simulations",
            description=(
                f"**ID {sim_id}**\n"
                f"Condition: `{cond}`\n"
                f"Seed: `{seed}`\n"
                f"Saved: {created}"
            ),
            color=0x3498DB
        )
        embed.set_footer(text=f"Page {self.current+1}/{len(self.rows)} (max 10 sims)")
        await interaction.response.edit_message(embed=embed, view=self)

    @nextcord.ui.button(label="⬅️ Prev", style=nextcord.ButtonStyle.secondary)
    async def prev(self, button: nextcord.ui.Button, interaction: Interaction):
        self.current = (self.current - 1) % len(self.rows)
        await self._update_embed(interaction)

    @nextcord.ui.button(label="Next ➡️", style=nextcord.ButtonStyle.secondary)
    async def nxt(self, button: nextcord.ui.Button, interaction: Interaction):
        self.current = (self.current + 1) % len(self.rows)
        await self._update_embed(interaction)

    @nextcord.ui.button(label="📂 Load", style=nextcord.ButtonStyle.primary)
    async def load(self, button: nextcord.ui.Button, interaction: Interaction):
        # replay the table simulation
        sim_id, cond, seed, _ = self.rows[self.current]
        # determine season from stored condition string
        season, _, rest = cond.partition("|")
        fixtures = fixture_list_2425 if season == "24/25" else fixture_list_2526
        promoted = PROMOTED[season]
        # re-seed RNGs so it's identical
        # (you may already have code to seed from `seed`)
        seed_int = int.from_bytes(seed.encode(), 'little') % (2**32)
        random.seed(seed_int); np.random.seed(seed_int); torch.manual_seed(seed_int)

        tbl, tries = simulate_until_vectorized(
            rest, fixtures,
            promoted=promoted,
            bias_factor=BIAS_FACTOR,
            max_sims=100000
        )
        # build the paginated embeds just like your original command
        pages = math.ceil(len(tbl) / 10)
        embeds = []
        for p in range(pages):
            chunk = tbl.iloc[p*10:(p+1)*10]
            e = nextcord.Embed(
                title=f"🔍 Loaded Simulation {season}",
                description=(f"Condition: `{rest}`\nSeed: `{seed}`\n"
                              f"Sims run: {tries}"),
                color=0x2ECC71
            )
            desc = "\n".join(
                f"**{pos}. {r.Team}** — {r.Pts} pts "
                f"({r.W}W {r.D}D {r.L}L, {r.GF}GF {r.GA}GA {r.GD}GD)"
                for pos, r in chunk.iterrows()
            )
            e.add_field(name="Standings", value=desc, inline=False)
            e.set_footer(text=f"Page {p+1}/{pages}")
            embeds.append(e)

        await interaction.response.send_message(
            embed=embeds[0],
            view=PaginatorView(embeds),
            ephemeral=True
        )

    @nextcord.ui.button(label="🗑️ Delete", style=nextcord.ButtonStyle.danger)
    async def delete(self, button: nextcord.ui.Button, interaction: Interaction):
        # delete from DB
        sim_id, _, _, _ = self.rows[self.current]
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM simulations WHERE id = ?", (sim_id,))
        conn.commit()
        conn.close()

        # remove locally and update or close
        self.rows.pop(self.current)
        if not self.rows:
            return await interaction.response.edit_message(
                content="🗑️ All simulations deleted.",
                embed=None,
                view=None
            )
        # clamp current
        self.current %= len(self.rows)
        await self._update_embed(interaction)







# ─── 11) /simulate gameweek (optional) ───────────────────────────────────────
@simulate_parent.subcommand(
    name="gameweek",
    description="Predict scores for a real 25/26 gameweek"
)
@cooldowns.cooldown(3, 10, bucket=cooldowns.SlashBucket.author)
async def simulate_gameweek(
    interaction: Interaction,
    gw: int = SlashOption(
        name="gw",
        description="Gameweek number (1–38)",
        required=True, min_value=1, max_value=38
    )
):
    original = await interaction.response.send_message(
        "<a:simulating:1383927364349726862> Predicting GW…", ephemeral=True
    )

    # pick new 25/26 DataFrame & its detected columns
    fx       = fx_2526
    home_c, away_c, week_c = (h26, a26, w26)
    promoted = PROMOTED["25/26"]

    # filter by the Round Number
    gw_df = fx[ fx[week_c] == gw ]
    if gw_df.empty:
        return await original.edit(content=f"❌ No fixtures found for GW{gw} 25/26")

    # now use your Dixon–Coles tensors for realistic λ/μ...
    preds = []
    for _, row in gw_df.iterrows():
        h, a = row[home_c], row[away_c]
        h = canonical_team(row[home_c])
        a = canonical_team(row[away_c])
        i, j = team_idx[h], team_idx[a]
        
        lam = α_t[i] * δ_t[j] * μh_t
        mu  = β_t[j] * γ_t[i] * μa_t
        if h in promoted: lam *= BIAS_FACTOR
        if a in promoted: mu  *= BIAS_FACTOR

        sh = int(torch.poisson(lam).item())
        sa = int(torch.poisson(mu).item())
        preds.append((h, a, sh, sa))

    # paginate in fives
    per_page = 5
    pages = math.ceil(len(preds)/per_page)
    embeds = []
    for p in range(pages):
        chunk = preds[p*per_page:(p+1)*per_page]
        e = nextcord.Embed(
            title=f"⚽ GW{gw} Predictions 25/26",
            color=0x9B59B6
        )
        for h,a,sh,sa in chunk:
            e.add_field(name=f"{h} vs {a}", value=f"**{sh}**–**{sa}**", inline=False)
        e.set_footer(text=f"Page {p+1}/{pages}")
        embeds.append(e)

    await original.edit(content=None, embed=embeds[0], view=PaginatorView(embeds))

TOP_GG_URL = "https://top.gg/bot/1383786497865678882/vote"

async def vote_callback(inter: nextcord.Interaction):
    """Fire when the user presses the Vote button."""
    await inter.response.send_message(
        f"🎉 Thanks for voting! {TOP_GG_URL}",
        ephemeral=True
    )
"""
        # 1) “finishes Nth”
        m = re.search(r'finish(?:es)?\s+(\d+)', text)
        if m:
            descs.append(("finish", team, int(m.group(1))))
            continue

        # 2) “wins league” ⇒ finish 1
        if re.search(r'\bwin(?:s)?\b.*\bleague\b', text):
            descs.append(("finish", team, 1))
            continue

        # 3) relegated / bottom ⇒ finish >= n_teams-2
        if "relegat" in text or "bottom" in text:
            descs.append(("finish_ge", team, n_teams-2))
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
"""

@bot.slash_command(name="custom-conditions", description="View custom conditions for /simulate custom-condition")
async def custom_conditions(interaction: Interaction):
    desc = """You can use the following conditions to simulate until a scenario is met:\n\n
            • `finish <N>`: Team finishes Nth in the table\n
            • `win league`: Team wins the league\n
            • `relegated`: Team is relegated (finishes in bottom 3)\n
            • `unbeaten`: Team goes unbeaten in the season\n
            • `<N> goals scored`: Team scores N+ goals in the season\n
            • `goals conceded <N>`: Team concedes fewer than N goals\n
            • `gets points <N>`: Team earns N+ points in the season\n
            • `goal difference <N>`: Team has a goal difference of at least N\n
            • `wins <n> games`: Team wins N number of games\n
            • `draws <n> games`: Team draws N number of games\n
            • `loses <n> games`: Team loses N number of games\n"""
    
    
    embed = nextcord.Embed(title="Custom Conditions for /simulate custom-condition", description=desc, color=nextcord.Colour.blurple())

    await interaction.response.send_message(embed=embed, ephemeral=True)

# ─── 12) /info ───────────────────────────────────────────────────────────────
@bot.slash_command(name="info", description="Information about the bot")
async def info(interaction: Interaction):
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = now - START_TIME
    days, rem = divmod(delta.total_seconds(), 86400)
    hrs, rem  = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)

    embed = nextcord.Embed(
        title=f"Hey there {interaction.user.display_name}! 👋",
        description=(
            "I'm your PL Simulator Bot — I use historical data & statistical models to "
            "simulate Premier League seasons and gameweeks.\n\n"
            "• /simulate table to get a full simulated season\n"
            "• /simulate gameweek to replay a real gameweek\n"
            "• /simulate custom-condition to loop until your scenario hits\n"
        ),
        color=nextcord.Colour.blurple()
    )
    embed.add_field(
        name="⌛ Uptime",
        value=f"{int(days)}d {int(hrs)}h {int(mins)}m {int(secs)}s",
        inline=False
    )
    embed.add_field(
        name="🛠️ Quick Links",
        value=(
            "Report a Bug - /bug • "
            "Suggest a Feature - /suggest • "
            "[Trello Board](https://trello.com/b/htvxO9KY/premier-league-sim-bot)"
        ),
        inline=False
    )    
    embed.set_footer(text="Developed by olxver.2025 | Version 1.1a")
    msg = await interaction.response.send_message(embed=embed, ephemeral=True)
    view = QuickButton(label="Vote for me on Top.gg!", emoji="✨", url="https://top.gg/bot/1383786497865678882/vote", color="link")
    await view.attach_to(msg)


# ─── 13) /changelog ──────────────────────────────────────────────────────────
@bot.slash_command(name="changelog", description="View bot updates and changelog")
async def changelog(interaction: Interaction):

    embeds = []
    for i, u in enumerate(bot.updates, start=1):
        e = nextcord.Embed(
            title=f"📦 version - {u['version']}",
            description="\n".join(f"- {c}" for c in u["changes"]),
            color=0x7289DA
        )
        e.set_footer(text=f"Page {i}/{len(bot.updates)}")
        embeds.append(e)

    await interaction.response.send_message(embed=embeds[0], view=PaginatorView(embeds), ephemeral=True)


# ─── 14) /bug & /suggest ─────────────────────────────────────────────────────
@bot.slash_command(name="bug", description="Report a bug or issue")
async def bug(interaction: Interaction):
    class BugReportModal(nextcord.ui.Modal):
        def __init__(self):
            super().__init__(title="🐛 Bug Report")
            self.add_item(nextcord.ui.TextInput(label="Description", style=nextcord.TextInputStyle.paragraph))
            self.add_item(nextcord.ui.TextInput(label="Steps to Reproduce", style=nextcord.TextInputStyle.paragraph))
            self.add_item(nextcord.ui.TextInput(label="Expected Behavior", style=nextcord.TextInputStyle.paragraph))
        async def callback(self, modal_inter: Interaction):
            await modal_inter.response.send_message("✅ Bug report submitted!", ephemeral=True)
            dev = bot.get_channel(1383918296717459457)
            if dev:
                await dev.send(
                    f"🐞 **Bug from {modal_inter.user.mention}:**\n"
                    f"**Desc:** {self.children[0].value}\n"
                    f"**Steps:** {self.children[1].value}\n"
                    f"**Expect:** {self.children[2].value}"
                )
    await interaction.response.send_modal(BugReportModal())

@bot.slash_command(name="suggest", description="Suggest a new feature or improvement")
async def suggest(interaction: Interaction):
    class SuggestionModal(nextcord.ui.Modal):
        def __init__(self):
            super().__init__(title="💡 Feature Suggestion")
            self.add_item(nextcord.ui.TextInput(label="Your Suggestion", style=nextcord.TextInputStyle.paragraph))
        async def callback(self, modal_inter: Interaction):
            await modal_inter.response.send_message("👍 Suggestion received!", ephemeral=True)
            dev = bot.get_channel(1383918296717459457)
            if dev:
                await dev.send(f"💭 **Suggestion from {modal_inter.user.mention}:**\n{self.children[0].value}")
    await interaction.response.send_modal(SuggestionModal())


@bot.event
async def on_guild_join(guild:nextcord.Guild):
    print(f"---------------------\njoined {guild.name} with {guild.member_count}, owner: {guild.owner}")
    chan = 1383918296717459457
    dev = bot.get_channel(chan)
    embed = nextcord.Embed(
        title="🤖 Bot Joined New Server"
        f"\n**Server:** {guild.name} ({guild.id})"
        f"\n**Owner:** {guild.owner} ({guild.owner.id if guild.owner else 'Unknown'})"
        f"\n**Members:** {guild.member_count}"
        )
    embed.colour = nextcord.Colour.green()
    embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
    embed.set_footer(text=f"Total servers: {len(bot.guilds)}")
    if dev:
        await dev.send(embed=embed)
    else:
        print(f"Failed to send join message to dev channel {chan}")
    

@bot.event
async def on_guild_remove(guild:nextcord.Guild):
    print(f"removed from guild {guild.name} with {guild.member_count}")
    chan = 1383918296717459457
    dev = bot.get_channel(chan)
    embed = nextcord.Embed(
        title="🤖 Bot removed from server"
        f"\n**Server:** {guild.name} ({guild.id})"
        f"\n**Owner:** {guild.owner} ({guild.owner.id if guild.owner else 'Unknown'})"
        f"\n**Members:** {guild.member_count}"
        )
    embed.colour = nextcord.Colour.red()
    embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
    embed.set_footer(text=f"Total servers: {len(bot.guilds)}")
    if dev:
        await dev.send(embed=embed)
    else:
        print(f"Failed to send leave message to dev channel {chan}")


    

class DebugControlView(nextcord.ui.View):
    def __init__(self, bot: commands.Bot, *, timeout: float | None = None):
        super().__init__(timeout=timeout)
        self.bot = bot

    @nextcord.ui.button(label="🔄 Restart", style=nextcord.ButtonStyle.danger)
    async def restart(self, button: nextcord.ui.Button, interaction: Interaction):
        await interaction.response.send_message("🔄 Restarting bot…", ephemeral=True)
        # this replaces the current process with a fresh one
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    @nextcord.ui.button(label="⏹️ Shutdown", style=nextcord.ButtonStyle.danger)
    async def shutdown(self, button: nextcord.ui.Button, interaction: Interaction):
        await interaction.response.send_message("⏹️ Shutting down…", ephemeral=True)
        await self.bot.close()

    @nextcord.ui.button(label="🟢 Status", style=nextcord.ButtonStyle.blurple)
    # allow changing of bot status
    async def status(self, button: nextcord.ui.Button, interaction: Interaction):
        async def status_modal():
            class StatusModal(nextcord.ui.Modal):
                def __init__(self):
                    super().__init__(title="Change Bot Status")
                    self.add_item(nextcord.ui.TextInput(
                        label="New Status",
                        placeholder="e.g. Playing PL Simulator",
                        required=True
                    ))

                async def callback(self, modal_inter: Interaction):
                    new_status = self.children[0].value
                    await bot.change_presence(activity=nextcord.Game(name=new_status))
                    await modal_inter.response.send_message(f"✅ Status changed to: {new_status}", ephemeral=True)

            await interaction.response.send_modal(StatusModal())

        await status_modal()


@bot.slash_command(name="debug", description="debug command for developers")
@application_checks.is_owner()
async def debug(interaction: nextcord.Interaction):
    original_message = await interaction.response.send_message(
        "<a:simulating:1383927364349726862>", ephemeral=True
    )

    # ----- 2) hardware ---------------------------------------------------
    sys_info = platform.system()
    cpu = platform.processor() or platform.machine()
    py  = platform.python_version()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
    vram = (
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        if torch.cuda.is_available() else "n/a"
    )
    sys_info = f"{sys_info} {platform.release()} ({platform.version()})"

    try:
        vm   = psutil.virtual_memory()
        mem  = f"{vm.total/1e9:.1f} GB"
        load = ", ".join(f"{x:.2f}" for x in os.getloadavg())
    except Exception:
        mem, load = "n/a", "n/a"

    # ----- 3) bot scope --------------------------------------------------
    guilds  = len(bot.guilds)
    members = sum(g.member_count or 0 for g in bot.guilds)

    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # benchmark 100k sims on 24/25
    fixtures = fixture_list_2425
    promoted = PROMOTED["24/25"]
    pts_all = simulate_batch(
        fixtures,
        promoted=promoted,
        bias_factor=BIAS_FACTOR,
        n_sims=100000,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sim_secs = time.perf_counter() - t0

    # ----- 4) embed ------------------------------------------------------
    embed = nextcord.Embed(
        title="🛠️ Premier League Simulator — Debug",
        colour=0xE67E22
    )
    embed.add_field(
        name="⏱️ 100,000-sim benchmark",
        value=f"{sim_secs:.2f} s",
        inline=False
    )
    embed.add_field(
        name="🖥️ Hardware",
        value=f"{sys_info} • {cpu} • {gpu}",
        inline=False
    )
    embed.add_field(name="💾 Memory",   value=f"{mem} • {vram} VRAM", inline=True)
    embed.add_field(name="📈 Load avg", value=load,                 inline=True)
    embed.add_field(name="🐍 Python",   value=py,                   inline=True)
    embed.add_field(name="🌐 Servers",  value=str(guilds),         inline=True)
    embed.add_field(name="👥 Members",  value=str(members),        inline=True)

    view = DebugControlView(bot)
    await original_message.edit(content=None, embed=embed, view=view)








# ─── FAQ PARENT ──────────────────────────────────────────────────────────────
@bot.slash_command(
    name="faq",
    description="Frequently Asked Questions about the PL Simulator"
)
async def faq_parent(interaction: Interaction):
    # This function body can stay empty, it's just a container for subcommands.
    pass


# ─── /faq seeds ──────────────────────────────────────────────────────────────
@faq_parent.subcommand(
    name="seeds",
    description="How simulation seeds control reproducibility"
)
async def faq_seeds(interaction: Interaction):
    embed = nextcord.Embed(
        title="🌱 FAQ: Simulation Seeds",
        color=0x00ADEF
    )
    embed.add_field(
        name="What is a seed?",
        value=(
            "Every time we run a simulation, we use a **seed** to initialize the random number "
            "generators (PyTorch, NumPy, Python). That means if you run the same simulation "
            "with the same **condition** and **seed**, you'll get **exactly** the same final table."
        ),
        inline=False
    )
    embed.add_field(
        name="How do I use it?",
        value=(
            "• To let the bot pick a random seed each run, simply omit the `seed` option:\n"
            "```\n"
            "/simulate custom-condition season:25/26 condition:\"Team X wins league\"\n"
            "```\n"
            "• To reproduce a previous result, copy the seed shown in the footer and pass it:\n"
            "```\n"
            "/simulate custom-condition season:25/26 condition:\"Team X wins league\" seed:abcdef1234567890\n"
            "```"
        ),
        inline=False
    )
    embed.add_field(
        name="Seed format",
        value=(
            "Seeds are simple hexadecimal strings (0–9, a–f), typically **16 characters** long. "
            "You can supply any hex string up to 16 chars; if it’s shorter we’ll pad internally."
        ),
        inline=False
    )
    embed.set_footer(
        text="After each sim, the seed is displayed so you can copy & reuse it."
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)


@faq_parent.subcommand(
    name="model",
    description="How the simulation model predicts outcomes"
)
async def faq_model(interaction: Interaction):
    embed = nextcord.Embed(
        title="⚽ FAQ: How the Model Predicts",
        color=0x00ADEF
    )

    embed.add_field(
        name="1. Fitting Team Strengths",
        value=(
            "We use a **Dixon–Coles Poisson model** on historical results:\n"
            "• Each team _i_ has an **attack** parameter αᵢ and a **defense** parameter βᵢ.\n"
            "• A low‐score correction ρ accounts for under-dispersion when both sides score ≤1.\n"
            "• We fit these via maximum likelihood over the past seasons’ home/away goals."
        ),
        inline=False
    )

    embed.add_field(
        name="2. Expected Goal Rates",
        value=(
            "For a match **Home (h)** vs **Away (a)**:\n"
            "• **λₕ = αₕ × δₐ × μₕ**  → home’s expected goals\n"
            "• **μₐ = βₐ × γₕ × μₐ**  → away’s expected goals\n"
            "• μₕ/μₐ are league-wide base rates and δ, γ adjust for home/away factors."
        ),
        inline=False
    )

    embed.add_field(
        name="3. Poisson Sampling",
        value=(
            "We draw goals:  \n"
            "```\n"
            "home_goals ~ Poisson(λₕ)  \n"
            "away_goals ~ Poisson(μₐ)\n"
            "```  \n"
            "Independently for each fixture (with a small Dixon–Coles tweak for 0–1 cases)."
        ),
        inline=False
    )

    embed.add_field(
        name="4. Season Simulation",
        value=(
            "We simulate **all 38 GWs** in parallel (GPU-accelerated!), tabulate W/D/L, GF/GA, Pts, then sort into a table.  \n"
            "Repeat **N** times (e.g. 100 000) to get finish-position probabilities, point distributions, etc."
        ),
        inline=False
    )

    embed.add_field(
        name="5. Promotion Bias",
        value=(
            "To account for newly promoted sides, we multiply their λ/μ by a **bias factor** (< 1) so they’re slightly under-rated initially."
        ),
        inline=False
    )

    embed.set_footer(
        text="All of this runs in milliseconds on a GPU, letting us explore thousands of scenarios!"
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)



@faq_parent.subcommand(
    name="custom-condition",
    description="Deep dive: how '/simulate custom-condition' works"
)
async def faq_custom_condition(interaction: Interaction):
    embed = nextcord.Embed(
        title="🔁 FAQ: /simulate custom-condition",
        color=0x00ADEF
    )

    embed.add_field(
        name="• What it does",
        value=(
            "Runs **up to 100 000** full‐season simulations in one GPU-vectorized batch, "
            "then stops at the **first** simulation that satisfies your condition(s).  "
            "You get that one realistic league table plus an **estimated probability** ≈ 1/tries."
        ),
        inline=False
    )

    embed.add_field(
        name="• Condition syntax",
        value=(
            "You write a mini-DSL of clauses, joined by `and`:\n"
            "```text\n"
            "finish <N>         → exact league finish (e.g. `finish 1`)\n"
            "win league         → same as `finish 1`\n"
            "relegated/bottom   → finish ≥18th (bottom 3)\n"
            "unbeaten           → zero losses in season\n"
            "<N> goals scored   → GF ≥ N\n"
            "conceded <N>       → GA < N\n"
            "<N> points         → Pts ≥ N\n"
            "goal difference <N>→ GD ≥ N\n"
            "wins/draws/loses N → exact W/D/L count\n"
            "```"
        ),
        inline=False
    )

    embed.add_field(
        name="• How it runs",
        value=(
            "1. **Parse** your input with regex and fuzzy‐matching to identify team names & metrics.\n"
            "2. **Simulate** all 20 teams × 38 GWs in one tensor: sample Poisson(homes_goals, away_goals).\n"
            "3. **Rank** each season by Pts → GD → GF, build a **ranks** tensor.\n"
            "4. **Mask** for your condition(s) and pick the **first** matching sim index.\n"
            "5. **Build** that one DataFrame, paginate into embeds, and show it."
        ),
        inline=False
    )

    embed.add_field(
        name="• Probability & timing",
        value=(
            "• **Estimated chance** = 1 / `tries`; displayed in the embed header.  \n"
            "• **Timing**: runs in under a second on GPU for 100 000 sims."
        ),
        inline=False
    )

    embed.add_field(
        name="• Reproducibility (Seeds)",
        value=(
            "You’ll see a **Seed** in the footer; reuse it with the `seed:` option to get **exactly** the same table again: (view /faq seeds for more)\n"
            "```text\n"
            "/simulate custom-condition season:25/26 \\\n"
            "    condition:\"Team X finish 4th\" seed:abcdef1234567890\n"
            "```"
        ),
        inline=False
    )

    embed.add_field(
        name="• Saving & viewing",
        value=(
            "After the sim you can click **💾 Save Simulation** to persist it to your personal history.  \n"
            "Use `/view simulations` to browse, **Load** any saved run, or **🗑️ Delete** it."
        ),
        inline=False
    )

    embed.set_footer(
        text="Pro tip: chain multiple clauses with 'and', e.g. “Man City win league and Man City get <40 GA”."
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)

@faq_parent.subcommand(
    name="commands",
    description="Full list of slash commands and what they do"
)
async def faq_commands(interaction: Interaction):
    embed = nextcord.Embed(
        title="📋 FAQ: Command Reference",
        color=0x00ADEF
    )

    embed.add_field(
        name="🔹 /simulate table",
        value="Simulate a full Premier League season (2025/26).",
        inline=False
    )
    embed.add_field(
        name="🔹 /simulate gameweek",
        value="Predict scores for a specific real gameweek (1–38) in 25/26.",
        inline=False
    )
    embed.add_field(
        name="🔹 /simulate custom-condition",
        value=(
            "Run up to 100,000 sims of the 25/26 season and stops when your scenario is met.  \n"
            "Options: `condition`, optional `seed` for reproducibility."
        ),
        inline=False
    )
    embed.add_field(
        name="🔹 /view simulations",
        value="Browse your saved `/simulate custom-condition` runs; Load or Delete any entry.",
        inline=False
    )
    embed.add_field(
        name="🔹 /info",
        value="Show bot uptime, quick links, and vote button.",
        inline=False
    )
    embed.add_field(
        name="🔹 /changelog",
        value="See recent updates and version history.",
        inline=False
    )
    embed.add_field(
        name="🔹 /bug",
        value="Report a bug.",
        inline=False
    )
    embed.add_field(
        name="🔹 /suggest",
        value="Suggest a feature or improvement.",
        inline=False
    )
    embed.add_field(
        name="🔹 /faq <topic>",
        value=(
            "Get detailed help on any topic.  \n"
            "Available topics: `seeds`, `model`, `custom-condition`, `commands` (this list)."
        ),
        inline=False
    )
    embed.set_footer(
        text="Use `/faq commands` to see this list again anytime!"
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.slash_command(name="user")
async def user(interaction: Interaction):
    pass
@user.subcommand(
    name="stats",
    description="View your simulation stats"
)
async def user_stats(interaction: Interaction):
    """Fetch and display summary statistics of the user's saved simulations."""
    user_id = str(interaction.user.id)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) Total saved simulations
    cur.execute(
        "SELECT COUNT(*) FROM simulations WHERE user_id = ?",
        (user_id,)
    )
    total = cur.fetchone()[0]

    # 2) Last saved timestamp
    cur.execute(
        "SELECT MAX(created_at) FROM simulations WHERE user_id = ?",
        (user_id,)
    )
    last_saved = cur.fetchone()[0] or "N/A"

    # 3) Top 3 most-used conditions
    cur.execute(
        """
        SELECT condition, COUNT(*) AS cnt
        FROM simulations
        WHERE user_id = ?
        GROUP BY condition
        ORDER BY cnt DESC
        LIMIT 3
        """,
        (user_id,)
    )
    top_conditions = cur.fetchall()  # list of (condition, cnt)
    conn.close()

    embed = nextcord.Embed(
        title="📊 Your Simulation Stats",
        color=0x3498DB
    )
    embed.add_field(
        name="Total simulations saved",
        value=str(total),
        inline=False
    )
    embed.add_field(
        name="Last saved at",
        value=last_saved,
        inline=False
    )

    if top_conditions:
        cond_lines = "\n".join(
            f"• `{cond}` — {cnt}x"
            for cond, cnt in top_conditions
        )
    else:
        cond_lines = "You haven’t saved any simulations yet."

    embed.add_field(
        name="Top saved conditions",
        value=cond_lines,
        inline=False
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.slash_command(
    name="global-stats",
    description="View overall simulation statistics"
)
async def global_stats(interaction: Interaction):
    """Show the most active user and most common condition."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    top_user_row = cur.fetchone()

    cur.execute(
        "SELECT condition, COUNT(*) AS c FROM global_sims GROUP BY condition ORDER BY c DESC LIMIT 1"
    )
    top_cond_row = cur.fetchone()
    conn.close()

    embed = nextcord.Embed(
        title="🌐 Global Simulation Stats",
        color=0x3498DB
    )

    if top_user_row:
        uid, cnt = top_user_row
        try:
            user = await bot.fetch_user(int(uid))
            name = user.name
        except Exception:
            name = f"User {uid}"
        embed.add_field(
            name="Top simulator",
            value=f"{name} — {cnt} sims",
            inline=False
        )
    else:
        embed.add_field(
            name="Top simulator",
            value="No simulations yet.",
            inline=False
        )

    if top_cond_row:
        cond, cnt = top_cond_row
        embed.add_field(
            name="Most popular condition",
            value=f"`{cond}` — {cnt} sims",
            inline=False
        )
    else:
        embed.add_field(
            name="Most popular condition",
            value="No simulations yet.",
            inline=False
        )

    await interaction.response.send_message(embed=embed, ephemeral=True)

USER_DB_PATH = "data/users.db"

class SettingsView(nextcord.ui.View):
    def __init__(self, user_id: str):
        super().__init__(timeout=None)
        self.user_id = user_id
        self._load_settings()
        self._update_buttons()

    def _load_settings(self):
        conn = sqlite3.connect(USER_DB_PATH)
        cur = conn.cursor()
        # ensure row exists
        cur.execute(
            "INSERT OR IGNORE INTO settings (user_id) VALUES (?)",
            (self.user_id,)
        )
        conn.commit()
        # load flags
        cur.execute(
            "SELECT auto_save, beta_mode FROM settings WHERE user_id = ?",
            (self.user_id,)
        )
        row = cur.fetchone()
        conn.close()
        self.auto_save_enabled = bool(row[0])
        self.beta_enabled      = bool(row[1])

    def _build_embed(self) -> nextcord.Embed:
        e = nextcord.Embed(title=f"⚙️ Your settings:", description=(
                f"Auto-save: {'✅ Enabled' if self.auto_save_enabled else '❌ Disabled'}\n"
                f"Beta mode: {'✅ Enabled' if self.beta_enabled      else '❌ Disabled'}"
            ),
            color=0x3498DB)
        user_url = bot.get_user(self.user_id)
        e.set_thumbnail(url=user_url.display_avatar.url if user_url else None)
        return e

    def _update_buttons(self):
        for btn in self.children:
            if btn.custom_id == "autosave":
                btn.style = (
                    nextcord.ButtonStyle.success
                    if self.auto_save_enabled else
                    nextcord.ButtonStyle.danger
                )
            elif btn.custom_id == "beta":
                btn.style = (
                    nextcord.ButtonStyle.success
                    if self.beta_enabled else
                    nextcord.ButtonStyle.danger
                )

    @nextcord.ui.button(
        label="Auto-save", style=nextcord.ButtonStyle.danger, custom_id="autosave"
    )
    async def autosave_btn(self, button: nextcord.ui.Button, interaction: Interaction):
        # flip flag
        self.auto_save_enabled = not self.auto_save_enabled
        # persist
        conn = sqlite3.connect(USER_DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            "UPDATE settings SET auto_save = ? WHERE user_id = ?",
            (1 if self.auto_save_enabled else 0, self.user_id)
        )
        conn.commit()
        conn.close()
        # update UI
        embed = self._build_embed()
        self._update_buttons()
        await interaction.response.edit_message(embed=embed, view=self)

    @nextcord.ui.button(
        label="Beta mode", style=nextcord.ButtonStyle.danger, custom_id="beta"
    )
    async def beta_btn(self, button: nextcord.ui.Button, interaction: Interaction):
        self.beta_enabled = not self.beta_enabled
        conn = sqlite3.connect(USER_DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            "UPDATE settings SET beta_mode = ? WHERE user_id = ?",
            (1 if self.beta_enabled else 0, self.user_id)
        )
        conn.commit()
        conn.close()
        embed = self._build_embed()
        self._update_buttons()
        await interaction.response.edit_message(embed=embed, view=self)






@user.subcommand(
    name="settings",
    description="View your user settings"
)
async def user_settings(interaction: Interaction):
    view = SettingsView(str(interaction.user.id))
    embed = view._build_embed()
    # send the panel with buttons
    await interaction.response.send_message(
        embed=embed,
        view=view,
        ephemeral=True
    )

bot.run("")

