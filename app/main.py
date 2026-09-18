from fastapi import FastAPI

app = FastAPI(title="PawLine")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

