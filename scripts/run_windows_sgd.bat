@ECHO OFF
ECHO Activating virtual environment...
CALL venv_new\Scripts\activate.bat
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Ensure venv_new exists and is set up correctly.
    GOTO :EOF
)

ECHO Starting SGD hyperparameter tuning...

SET OPTIMIZER_NAME=sgd

FOR %%d IN (acsincome) DO (
    FOR %%o IN ( erm superquantile esrm) DO (
        ECHO.
        ECHO Running SGD with:
        ECHO   Dataset:   %%d
        ECHO   Objective: %%o
        ECHO   Optimizer: %OPTIMIZER_NAME%
        ECHO.
        python scripts/train.py --dataset %%d --objective %%o --optimizer %OPTIMIZER_NAME% --n_jobs 8 --n_epochs 128 --dataset_length 4000
        IF ERRORLEVEL 1 (
            ECHO Execution failed for Dataset: %%d, Objective: %%o, Optimizer: %OPTIMIZER_NAME%
        )
    )
)

ECHO.
ECHO SGD Script finished.
ECHO Deactivating virtual environment...
CALL venv_new\Scripts\deactivate.bat

:EOF 