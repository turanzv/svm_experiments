# SVM Analysis Notebooks

All scripts expect logs to be in a folder named "logs" in the root directoy of the project. The log for every experiment is then found in a folder describing the experiment:
- For execution experiments:
  - 128GB
  - 256GB
  - 512GB
  - 1TB
  - 1_5TB
- For program cache experiments:
  - 512PC
  - 1024PC
  - 2048PC

There are two types of logs:
- validator logs
- memory tracking logs

**MEM analysis_memory** holds the notebooks generating memory usage graphs. The notebooks ending in _simple come from an original simpler memory script that did not track page faults.

**MEM analysis_tps** holds the notebooks generating tps graphs.

The same naming convention is used for **PC analysis_memory** and **PC analysis_tps**. The notebooks in **PC anylsis_tps** also generates graphs on progarm cache performance.

**PC_data.ipynb** generates the graph of overall program cache performance as the cache gets larger.