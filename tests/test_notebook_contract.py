from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from IPython.core.inputtransformer2 import TransformerManager


NOTEBOOK = Path(__file__).parents[1] / "02_otimizacao_hiperparametros_ag.ipynb"


def notebook_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )


def notebook_cell(containing: str) -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        source = "".join(cell.get("source", []))
        if containing in source:
            return source
    raise AssertionError(f"Célula não encontrada: {containing}")


def test_notebook_contains_required_genetic_operators():
    source = notebook_source()

    assert "tools.selTournament" in source
    assert "tools.cxTwoPoint" in source
    assert "tools.mutUniformInt" in source
    assert "elite_size" in source


def test_all_notebook_code_cells_have_valid_python_syntax():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    transformer = TransformerManager()
    for index, cell in enumerate(payload.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        transformed = transformer.transform_cell(source)
        compile(transformed, f"notebook-cell-{index}", "exec")


def test_three_experiments_use_distinct_seeds_and_generations():
    source = notebook_source()

    assert "seed=RANDOM_STATE" in source
    assert "seed=RANDOM_STATE + 1" in source
    assert "seed=RANDOM_STATE + 2" in source
    assert "n_generations=10" in source
    assert "n_generations=12" in source
    assert "n_generations=15" in source


def test_fitness_uses_stratified_cross_validation():
    source = notebook_source()

    assert "StratifiedKFold" in source
    assert "cross_validate" in source
    assert "CV_SPLITS = 3" in source
    assert 'cv=CV_STRATEGY' in source


def test_search_space_includes_baseline_reference():
    source = notebook_source()

    assert "MAX_DEPTH_VALUES = [None] + list(range(3, 21))" in source
    assert "BASELINE_GENES = [400, 0, 2, 2, 0]" in source
    assert "population[0] = creator.Individual(BASELINE_GENES)" in source


def test_notebook_has_logging_and_colab_detection():
    source = notebook_source()

    assert 'RUNNING_IN_COLAB = "google.colab" in sys.modules' in source
    assert "logging.FileHandler" in source
    assert "execucao.log" in source


def test_prompt_explicitly_handles_missing_values_and_calibration():
    source = notebook_source()

    assert 'return "não informado"' in source
    assert "não foi calibrada para representar risco clínico" in source


def test_prompt_formats_missing_value_without_nan():
    namespace = {"pd": pd, "np": np}
    exec(notebook_cell("def formatar_valor_prompt"), namespace)

    prompt = namespace["criar_prompt_interpretacao"](
        {
            "classe_predita": "negative",
            "probabilidade_negative": 0.6,
            "probabilidade_hypothyroid": 0.4,
            "principais_variaveis": [{"variavel": "age", "valor": np.nan}],
        },
        {
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.95,
            "f1": 0.87,
            "auc": 0.98,
        },
    )

    assert "age: não informado" in prompt
    assert "age: nan" not in prompt
    assert "risco clínico" in prompt


def test_genetic_algorithm_preserves_elite_and_gene_bounds():
    import random
    import time

    fitness_cache = {}
    cache_stats = {"acertos": 0, "novas_avaliacoes": 0}

    class SilentLogger:
        def info(self, *args, **kwargs):
            return None

    bounds = {
        "n_estimators": (2, 8),
        "max_depth": [None, 3, 4, 5, 6],
        "min_samples_split": (2, 5),
        "min_samples_leaf": (1, 4),
        "max_features": ["sqrt", "log2"],
    }

    def evaluate_individual(individual):
        nonlocal fitness_cache, cache_stats
        key = tuple(int(gene) for gene in individual)
        if key in fitness_cache:
            cache_stats["acertos"] += 1
            individual.metrics = fitness_cache[key]["metrics"].copy()
            return fitness_cache[key]["fitness"]
        cache_stats["novas_avaliacoes"] += 1
        recall = sum(individual) / 30.0
        f1 = recall * 0.9
        auc = recall * 0.8
        metrics = {
            "accuracy": recall,
            "precision": f1,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }
        individual.metrics = metrics.copy()
        fitness = (recall, f1, auc)
        fitness_cache[key] = {"fitness": fitness, "metrics": metrics}
        return fitness

    namespace = {
        "np": np,
        "random": random,
        "time": time,
        "base": base,
        "creator": creator,
        "tools": tools,
        "algorithms": algorithms,
        "PARAM_BOUNDS": bounds,
        "BASELINE_GENES": [4, 0, 2, 2, 0],
        "RANDOM_STATE": 42,
        "fitness_cache": fitness_cache,
        "cache_estatisticas": cache_stats,
        "evaluate_individual": evaluate_individual,
        "logger": SilentLogger(),
    }
    exec(notebook_cell("def run_genetic_algorithm"), namespace)

    best, _, logbook, execution = namespace["run_genetic_algorithm"](
        pop_size=10,
        n_generations=4,
        crossover_prob=0.7,
        mutation_prob=0.3,
        seed=42,
        elite_size=1,
    )

    maxima = list(logbook.select("max"))
    assert maxima == sorted(maxima)
    assert len(logbook) == 5
    assert execution["elite_size"] == 1
    assert bounds["n_estimators"][0] <= best[0] <= bounds["n_estimators"][1]
    assert 0 <= best[1] < len(bounds["max_depth"])
    assert bounds["min_samples_split"][0] <= best[2] <= bounds["min_samples_split"][1]
    assert bounds["min_samples_leaf"][0] <= best[3] <= bounds["min_samples_leaf"][1]
    assert 0 <= best[4] < len(bounds["max_features"])
