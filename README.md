# AI AWS Demo Streamlit

## Build and Run

```bash
docker build -t demo_streamlit_segmentation_service -f Dockerfile .
```

```bash
docker run -p 8081:8501 -e IP=XXX -e PORT=XXX -e BUCKET=XXX -e -ACCESS_KEY=XXX -e SECRET_KEY=XXX demo_streamlit_segmentation_service
```