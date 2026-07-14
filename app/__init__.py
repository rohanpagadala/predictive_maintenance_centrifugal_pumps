"""FastAPI serving layer for the Predictive Maintenance project."""

import warnings

# Same suppression the training notebook uses -- a pre-existing pandas
# groupby.apply deprecation notice from pdm_utils.handle_missing_values,
# not something this app introduces. Left untouched per "do not modify the
# existing ML pipeline"; silenced here instead of in pdm_utils.py.
warnings.filterwarnings("ignore", category=FutureWarning)
