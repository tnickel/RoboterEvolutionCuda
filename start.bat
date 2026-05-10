@echo off
cd /d "%~dp0"
title RoboterEvolution CUDA Launcher
color 0B

echo ========================================================
echo        RoboterEvolution - CUDA Training System
echo ========================================================
echo.
echo Willkommen im Neuro-Oekosystem!
echo.
echo Die Anwendung startet gleich...
echo Das Config-Menue bietet dir folgende Moeglichkeiten:
echo - Konfiguration anpassen
echo - Hall of Fame ansehen
echo - Training (Headless) fuer maximale CUDA-Performance
echo - Test (Visuell) zum Beobachten der Evolution
echo.
echo ========================================================
echo.

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    color 0C
    echo [FEHLER] Das Programm wurde unerwartet beendet.
    echo Bitte pruefe die Fehlermeldung oben.
    pause
)
