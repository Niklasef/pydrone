Remove-Item -Path ./gym-output/current/* -Recurse -Force
python.exe .\main.py k --gym --no-render | Tee-Object -FilePath ./gym-output/current/gym.log
python.exe .\benchmark.py
