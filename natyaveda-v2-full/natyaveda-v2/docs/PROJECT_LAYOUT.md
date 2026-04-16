# Project Layout Guide

This document defines a cleaner and more maintainable folder layout for the NatyaVeda v2 workspace.

## Core folders

- src/: application source code
- scripts/: command-line entry scripts
- config/: runtime and training configuration
- docs/: documentation and architecture notes
- tests/: automated test suite
- notebooks/: analysis notebooks intended to be kept

## Data and model artifacts

- data/: dataset storage and generated splits
- outputs/: generated inference outputs and temporary run outputs
- reports/: generated evaluation and analysis reports
- weights/: model checkpoints and pretrained files

## New organization improvements

The following folders were introduced to reduce top-level clutter:

- archive/versions/: moved loose version marker files
- archive/notes/: moved ad-hoc report notes
- experiments/scratch/: moved temporary notebook and temporary smoke script
- logs/: centralized runtime logs

## Practical conventions

- Keep only reusable notebooks under notebooks/.
- Keep temporary experiments in experiments/scratch/.
- Keep one-off notes in archive/notes/.
- Keep generated logs out of long-term source history.
