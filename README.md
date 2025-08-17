# Premier League Bot

This repository contains a set of scripts for simulating English Premier League seasons and interacting with the results via a Discord bot.

The project loads historical match data, estimates team strength parameters and uses them to produce match predictions. A Nextcord based bot allows users to fetch predictions inside Discord.

## Setup

1. Ensure Python 3.10+ is installed.
2. (Optional) create and activate a virtual environment.
3. Install required packages:

   ```bash
   pip install pandas numpy torch nextcord cooldowns scipy scikit-learn kaggle
   ```

4. Download the Premier League dataset from Kaggle:

   ```bash
   python data_download.py
   ```

   The data will be extracted to the `data/` directory.

## Main commands

- `python bot.py` – start the Discord bot.
- `python train_model.py` – train the season prediction model.
- `python prepare_season_table.py` – build aggregated season tables from match data.

## Project layout

- `bot.py` – Nextcord bot for Discord.
- `data/` – raw and processed datasets as well as model parameters.
- Additional scripts such as `seed_generation.py` and `25_26_table.py` provide utilities for fixture processing and reproducible simulations.

## License

This project is open source under the MIT license. See [LICENSE](LICENSE) for details.

