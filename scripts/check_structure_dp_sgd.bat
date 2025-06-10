@ECHO OFF
ECHO Activating virtual environment...
CALL venv_new\Scripts\activate.bat
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Ensure venv_new exists and is set up correctly.
    GOTO :EOF
)

ECHO Starting DP-SGD structure check...

SET dataset=acsincome
SET objective=extremile
SET n_epochs=2
SET n_jobs=1
SET common_args=--dataset %dataset% --objective %objective% --n_epochs %n_epochs% --n_jobs %n_jobs% --dataset_length 4000

REM --- DP_SGD Runs ---
SET optimizer_dp=dp_sgd
SET lr_dp=1e-3
SET bs_dp=64
SET ct_dp=1.0

FOR %%e IN (2 10 1000000) DO (
    ECHO.
    ECHO Running DP_SGD with Epsilon: %%e
    python scripts/train.py %common_args% --optimizer %optimizer_dp% --epsilon %%e --single_lr %lr_dp% --single_batch_size %bs_dp% --single_clip_threshold %ct_dp%
    IF ERRORLEVEL 1 (
        ECHO DP_SGD Execution failed for Epsilon: %%e
    )
)

ECHO.
ECHO DP-SGD Structure check script finished.
ECHO Deactivating virtual environment...
CALL venv_new\Scripts\deactivate.bat

:EOF 