# ILDARS_revised

The ildars module contains all the actual computation of the ildars pipeline, e.g. clustering, wall normal vector computation etc.
The evaluation module contains all the "meta" code for simulating the measurements and evaluating the algorithms of the ildars pipeline.

Decencies are managed using pipenv.
* run `pipenv update` to install dependencies.
* run `pipenv run python3 -m evaluation` to evaluate all implemented algorithms.
