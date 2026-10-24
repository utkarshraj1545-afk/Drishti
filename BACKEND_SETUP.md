# DRISHTI backend setup (Windows PowerShell)

## 1. Create and activate a virtual environment
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configure Google Earth Engine credentials
Generate/rotate the service-account key in your cloud console. Store the JSON key outside this repository, for example:
`C:\Users\YOUR_USER\earth-engine-service-account.json`

Set environment variables in the same PowerShell window (replace the values):
```powershell
$env:EARTH_ENGINE_SERVICE_ACCOUNT = "your-service-account@your-project.iam.gserviceaccount.com"
$env:EARTH_ENGINE_KEY_PATH = "C:\Users\YOUR_USER\earth-engine-service-account.json"
```

Never commit the key, put it in a shared ZIP, or send it to another AI tool.

## 3. Start the API
Run from the `Drishti` project root:
```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/health` and confirm the response is `{"status":"healthy"}`.
Interactive API docs: `http://127.0.0.1:8000/docs`.

## 4. Test
Start with `/health`, then use `/docs` to test `POST /analyze/change` with a small AOI and date ranges known to contain imagery. PDF export uses `POST /analyze/change/pdf`.
