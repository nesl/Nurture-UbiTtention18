#!/bin/bash

while true; do
	clear
	date
	python3 emulator/main_detect_abusive_workers.py
	sleep 5
done
