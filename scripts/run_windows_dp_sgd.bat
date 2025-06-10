@ECHO OFF
ECHO Activating virtual environment...
CALL venv_new\Scripts\activate.bat
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Ensure venv_new exists and is set up correctly.
    GOTO :EOF
)

ECHO Starting DP-SGD hyperparameter tuning...

REM Define your datasets, objectives, and epsilons here
SET DATASETS_TO_RUN=acsincome
SET OBJECTIVES_TO_RUN=erm esrm superquantile
SET OPTIMIZERS_TO_RUN=dp_sgd
SET EPSILONS_TO_RUN=2.0 4.0 10.0 100000.0

FOR %%d IN (%DATASETS_TO_RUN%) DO (
    FOR %%o IN (%OBJECTIVES_TO_RUN%) DO (
        FOR %%p IN (%OPTIMIZERS_TO_RUN%) DO (
            FOR %%e IN (%EPSILONS_TO_RUN%) DO (
                CALL :RUN_TRAIN %%d %%o %%p %%e
            )
        )
    )
)

ECHO.
ECHO DP-SGD Script finished.
ECHO Deactivating virtual environment...
CALL venv_new\Scripts\deactivate.bat
GOTO :EOF

:RUN_TRAIN
SETLOCAL
SET CURRENT_DATASET=%1
SET CURRENT_OBJECTIVE=%2
SET CURRENT_OPTIMIZER=%3
SET CURRENT_EPSILON=%4

ECHO.
ECHO Running DP-SGD with:
ECHO   Dataset:   %CURRENT_DATASET%
ECHO   Objective: %CURRENT_OBJECTIVE%
ECHO   Optimizer: %CURRENT_OPTIMIZER%
ECHO   Epsilon:   %CURRENT_EPSILON%
ECHO.

python scripts/train.py --dataset %CURRENT_DATASET% --objective %CURRENT_OBJECTIVE% --optimizer %CURRENT_OPTIMIZER% --n_jobs 8 --n_epochs 128 --dataset_length 4000 --epsilon %CURRENT_EPSILON%

IF ERRORLEVEL 1 (
    ECHO Execution failed for Dataset: %CURRENT_DATASET%, Objective: %CURRENT_OBJECTIVE%, Optimizer: %CURRENT_OPTIMIZER%, Epsilon: %CURRENT_EPSILON%
) ELSE (
    ECHO Successfully completed for Dataset: %CURRENT_DATASET%, Objective: %CURRENT_OBJECTIVE%, Optimizer: %CURRENT_OPTIMIZER%, Epsilon: %CURRENT_EPSILON%
)
ENDLOCAL
GOTO :EOF 