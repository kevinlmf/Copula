from copula_modeling.pipeline import run_copula_pipeline

if __name__ == "__main__":
    print("🚀 Starting Copula Modeling Pipeline...")
    model, params, summary = run_copula_pipeline()
    print("📊 Summary:", summary)
