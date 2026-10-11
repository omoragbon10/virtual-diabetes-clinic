# Virtual Diabetes Clinic Triage API

This ML service predicts short-term disease progression based on patient check-in data (vitals, labs, lifestyle notes) to prioritize follow-up calls. 

## Local Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python src/train.py --version v0.1`
3. Run the API: `uvicorn app.main:app --port 80`

## Docker Usage
Pull and run the containerized service:
```bash
docker pull ghcr.io/<omoragbon10>/virtual-diabetes-clinic:v0.1
docker run -it -p 80:80 ghcr.io/<omoragbon10>/virtual-diabetes-clinic:v0.1