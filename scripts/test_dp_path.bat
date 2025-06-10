@ECHO OFF
ECHO Activating virtual environment...
CALL venv_new\Scripts\activate.bat
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Ensure venv_new exists and is set up correctly.
    GOTO :EOF
)

ECHO Starting DP-SGD path test...

SET dataset=acsincome
SET objective=extremile
SET optimizer=dp_sgd
SET epsilon=2
SET n_epochs=2
SET n_jobs=1

ECHO.
ECHO Running with:
ECHO   Dataset:   %dataset%
ECHO   Objective: %objective%
ECHO   Optimizer: %optimizer%
ECHO   Epsilon:   %epsilon%
ECHO   N_epochs:  %n_epochs%
ECHO.

python scripts/train.py --dataset %dataset% --objective %objective% --optimizer %optimizer% --n_jobs %n_jobs% --n_epochs %n_epochs% --dataset_length 4000 --epsilon %epsilon%

IF ERRORLEVEL 1 (
    ECHO Execution failed for Dataset: %dataset%, Objective: %objective%, Optimizer: %optimizer%, Epsilon: %epsilon%
) ELSE (
    ECHO Test completed successfully.
)

ECHO Deactivating virtual environment...
CALL venv_new\Scripts\deactivate.bat

ECHO Script finished.
:EOF 