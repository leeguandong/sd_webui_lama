import launch

if not launch.is_installed("lama_cleaner"):
    try:
        launch.run_pip("install lama-cleaner", "requirements for lama_cleaner")
    except Exception:
        print("Can't install lama-cleaner. Please follow the readme to install manually")
