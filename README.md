LANLEarthquakePrediction
==============================

https://www.kaggle.com/c/LANL-Earthquake-Prediction#description

Description
------------
Forecasting earthquakes is one of the most important problems in Earth science because of their devastating
consequences. Current scientific studies related to earthquake forecasting focus on three key points: when the event 
will occur, where it will occur, and how large it will be.

In this competition, you will address when the earthquake will take place. Specifically, you’ll predict the time
remaining before laboratory earthquakes occur from real-time seismic data.

If this challenge is solved and the physics are ultimately shown to scale from the laboratory to the field, researchers
will have the potential to improve earthquake hazard assessments that could save lives and billions of dollars in
infrastructure.

This challenge is hosted by Los Alamos National Laboratory which enhances national security by ensuring the safety of
the U.S. nuclear stockpile, developing technologies to reduce threats from weapons of mass destruction, and solving 
problems related to energy, environment, infrastructure, health, and global security concerns.

Evaluation
------------

Submissions are evaluated using the mean absolute error between the predicted time remaining before the next lab
earthquake and the act remaining time.

Submission File

For each seg_id in the test set folder, you must predict time_to_failure, which is the remaining time before the next 
lab earthquake. The file should contain a header and have the following format:

```
seg_id,time_to_failure
seg_00030f,0
seg_0012b5,0
seg_00184e,0
...
```

Timeline
------------

* May 27, 2019 - Entry deadline. You must accept the competition rules before this date in order to compete.

* May 27, 2019 - Team Merger deadline. This is the last day participants may join or merge teams.

* May 27, 2019 - External Data Disclosure deadline. All external data used in the competition must be disclosed in the
forums by this date.

* June 3, 2019 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve
the right to update the contest timeline if they deem it necessary.


Prizes
------------

* 1st Place- $20,000
* 2nd Place- $15,000
* 3rd Place- $7,000
* 4th Place- $5,000
* 5th Place- $3,000

Additional information
------------

The data are from an experiment conducted on rock in a double direct shear geometry subjected to bi-axial loading, a 
classic laboratory earthquake model (https://www.nature.com/articles/ncomms11104)

Two fault gouge layers are sheared simultaneously while subjected to a constant normal load and a prescribed shear 
velocity. The laboratory faults fail in repetitive cycles of stick and slip that is meant to mimic the cycle of loading 
and failure on tectonic faults. While the experiment is considerably simpler than a fault in Earth, it shares many 
physical characteristics.

Los Alamos' initial work (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL074677) showed that the 
prediction of laboratory earthquakes from continuous seismic data is possible in the case of quasi-periodic laboratory 
seismic cycles. In this competition, the team has provided a much more challenging dataset with considerably more 
aperiodic earthquake failures.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
