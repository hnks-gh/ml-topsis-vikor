"""Quick test for pipeline v2."""
import sys
sys.path.insert(0, '.')

from src.data_loader import PanelDataLoader
from src.pipeline_v2 import PipelineV2

# Generate test data
loader = PanelDataLoader()
panel_data = loader.generate_synthetic(n_provinces=5, n_years=4, n_components=3)

# Run pipeline with verbose output
pipeline = PipelineV2(output_dir='outputs_test', ml_mode='fast', verbose=True)
result = pipeline.run(panel_data)

# Print results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Traditional MCDM methods: {list(result.traditional_mcdm.keys())}")
print(f"Fuzzy MCDM methods: {list(result.fuzzy_mcdm.keys())}")
print(f"ML Forecast available: {result.ml_forecast is not None}")
print(f"Execution time: {result.execution_time:.2f}s")
