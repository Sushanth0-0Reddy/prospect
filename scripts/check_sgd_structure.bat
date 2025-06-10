@ECHO OFF
ECHO Activating virtual environment...
CALL venv_new\Scripts\activate.bat
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Ensure venv_new exists and is set up correctly.
    GOTO :EOF
)

ECHO Starting SGD structure check...

SET dataset=acsincome
SET objective=extremile
SET n_epochs=2
SET n_jobs=1
SET common_args=--dataset %dataset% --objective %objective% --n_epochs %n_epochs% --n_jobs %n_jobs% --dataset_length 4000

REM --- SGD Run ---
SET optimizer_sgd=sgd
SET lr_sgd=1e-4
SET bs_sgd=128

ECHO.
ECHO Running SGD with fixed parameters...
ECHO   Dataset:   %dataset%
ECHO   Objective: %objective%
ECHO   Optimizer: %optimizer_sgd%
ECHO   LR:        %lr_sgd%
ECHO   Batch Size:%bs_sgd%
ECHO.

python scripts/train.py %common_args% --optimizer %optimizer_sgd% --single_lr %lr_sgd% --single_batch_size %bs_sgd%

IF ERRORLEVEL 1 (
    ECHO SGD Execution failed.
) ELSE (
    ECHO SGD Execution completed. Check output in hp_tuning_experiments\results_sgd\%dataset%
)

ECHO.
ECHO SGD structure check script finished.
ECHO Deactivating virtual environment...
CALL venv_new\Scripts\deactivate.bat

:EOF 