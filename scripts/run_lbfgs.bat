@ECHO OFF
ECHO Activating virtual environment...
CALL venv_new\Scripts\activate.bat
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Ensure venv_new exists and is set up correctly.
    GOTO :EOF
)

ECHO Running L-BFGS for multiple objectives...

SET dataset=acsincome
SET output_dir=hp_tuning_experiments/results_lbfgs

REM Define the list of objectives to run
SET objectives_to_run=extremile superquantile esrm erm

ECHO.
ECHO Parameters:
ECHO   Dataset:    %dataset%
ECHO   Output Dir: %output_dir%
ECHO   Objectives: %objectives_to_run%
ECHO.

FOR %%o IN (%objectives_to_run%) DO (
    ECHO #####################################################
    ECHO Running L-BFGS for Objective: %%o
    ECHO #####################################################
    python scripts/lbfgs.py --dataset %dataset% --objective %%o --output_base_dir %output_dir%
    IF ERRORLEVEL 1 (
        ECHO L-BFGS Execution failed for Dataset: %dataset%, Objective: %%o
    ) ELSE (
        ECHO L-BFGS Execution completed for Objective: %%o. Results in %output_dir%\%dataset%\
    )
    ECHO.
)

ECHO.
ECHO All L-BFGS runs finished.
ECHO Deactivating virtual environment...
CALL venv_new\Scripts\deactivate.bat

:EOF 