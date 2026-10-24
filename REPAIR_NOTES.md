# Repair notes

- Added backend dependencies to `requirements.txt`.
- Changed Earth Engine key configuration to environment variables, with a predictable local fallback path.
- Added clear missing-key validation before starting Earth Engine.
- Added strict YYYY-MM-DD date validation.
- Added `.gitignore` protections for credentials, virtual environments, and Python cache files.
- Added `.env.example` and `BACKEND_SETUP.md`.
- Excluded `.git`, virtual environments, Python bytecode, and real credential files from this ZIP.
- The API has not been live-tested against Google Earth Engine in this environment. After installing dependencies and configuring a rotated key locally, run the checks in `BACKEND_SETUP.md`.
