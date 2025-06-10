@ECHO OFF
ECHO Activating virtual environment...
CALL venv_new\Scripts\activate.bat
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Ensure venv_new exists and is set up correctly.
    GOTO :EOF
)

ECHO Starting performance comparison...

SET output_base=performance_check_experiments
SET dataset=acsincome
SET objective=extremile
SET n_epochs=64 
SET n_jobs=1 
SET dataset_length=4000

REM --- Single Hyperparameters ---
SET lr_single=1e-4
SET bs_single=128
SET ct_single=1.0 
SET eps_single=4.0

SET common_args=--dataset %dataset% --objective %objective% --n_epochs %n_epochs% --n_jobs %n_jobs% --dataset_length %dataset_length% --output_base_dir %output_base%

REM --- SGD Run ---
ECHO.
ECHO ###################################
ECHO Running SGD Performance Check...
ECHO ###################################
ECHO   Optimizer: sgd
ECHO   LR:        %lr_single%
ECHO   Batch Size:%bs_single%
ECHO.
python scripts/train.py %common_args% --optimizer sgd --single_lr %lr_single% --single_batch_size %bs_single%
IF ERRORLEVEL 1 (
    ECHO SGD Execution failed.
)

REM --- DP-SGD Run ---
ECHO.
ECHO ###################################
ECHO Running DP-SGD Performance Check...
ECHO ###################################
ECHO   Optimizer: dp_sgd
ECHO   LR:        %lr_single%
ECHO   Batch Size:%bs_single%
ECHO   Clip Thr:  %ct_single%
ECHO   Epsilon:   %eps_single%
ECHO.
python scripts/train.py %common_args% --optimizer dp_sgd --epsilon %eps_single% --single_lr %lr_single% --single_batch_size %bs_single% --single_clip_threshold %ct_single%
IF ERRORLEVEL 1 (
    ECHO DP-SGD Execution failed.
)

ECHO.
ECHO Performance comparison script finished.
ECHO Deactivating virtual environment...
CALL venv_new\Scripts\deactivate.bat

:EOF 