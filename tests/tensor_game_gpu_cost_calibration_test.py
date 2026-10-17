import importlib.util
from pathlib import Path

MODULE=Path(__file__).parents[1]/"scripts/benchmark_tensor_game_gpu_cost_calibration.py"
spec=importlib.util.spec_from_file_location("gpu_cost_calibration",MODULE); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
ANALYSIS=Path(__file__).parents[1]/"scripts/analyze_tensor_game_gpu_cost_calibration.py"
analysis_spec=importlib.util.spec_from_file_location("gpu_cost_analysis",ANALYSIS); analysis=importlib.util.module_from_spec(analysis_spec); analysis_spec.loader.exec_module(analysis)

def test_distribution_and_percentile():
    result=module.distribution([1.0,2.0,3.0,4.0])
    assert result["median_ms"]==2.5 and result["iqr_ms"]==1.5 and result["cv"]>0

def test_bootstrap_is_seeded_and_bounded():
    first=module.bootstrap_median_ci([.9,1.0,1.1],7,200)
    assert first==module.bootstrap_median_ci([.9,1.0,1.1],7,200)
    assert .9<=first[0]<=first[1]<=1.1

def test_spearman_supports_ties_and_direction():
    assert analysis.spearman([1,2,3],[3,2,1])==-1
    assert 0<analysis.spearman([1,1,2],[1,2,3])<1
