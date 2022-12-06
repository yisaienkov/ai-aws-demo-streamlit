# AI AWS Demo Streamlit

## Build and Run

```bash
docker build -t demo_streamlit_segmentation_service -f Dockerfile .
```

```bash
docker run -p 8081:8501 -e IP=XXX -e PORT=XXX -e BUCKET=XXX -e AWS_ACCESS_KEY_ID=XXX -e AWS_SECRET_ACCESS_KEY=XXX demo_streamlit_segmentation_service
```
