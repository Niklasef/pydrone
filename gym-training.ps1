Remove-Item -Path ./gym-output/current/* -Recurse -Force
python.exe .\main.py --gym --gym-time 200 --no-render | Tee-Object -FilePath ./gym-output/current/gym.log
python.exe .\benchmark.py
