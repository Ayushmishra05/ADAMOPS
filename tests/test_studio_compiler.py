"""
Tests for the AdamOps Studio Compiler engine.
Verifies DAG-to-Python conversion dynamically.
"""

import ast
import tempfile
import pytest

from adamops.studio.compiler import compile_pipeline


def test_compile_pipeline_empty():
    with pytest.raises(Exception, match="Cannot compile an empty pipeline"):
        compile_pipeline({"nodes": [], "connections": []})


def test_compile_pipeline_valid_ast():
    """
    Test that the compiler yields perfectly valid Python abstract syntax trees
    meaning the code holds no indentation or syntax parse errors.
    """
    pipeline_data = {
        "nodes": [
            {
                "id": "node_1",
                "type": "load_csv",
                "params": {"filepath": "dummy.csv"}
            },
            {
                "id": "node_2",
                "type": "train_test_split",
                "params": {"target": "label", "test_size": 0.2}
            },
            {
                "id": "node_3",
                "type": "train_classification",
                "params": {"algorithm": "random_forest"}
            }
        ],
        "connections": [
            {"from_node": "node_1", "from_port": "dataframe", "to_node": "node_2", "to_port": "dataframe"},
            {"from_node": "node_2", "from_port": "X_train", "to_node": "node_3", "to_port": "X_train"},
            {"from_node": "node_2", "from_port": "y_train", "to_node": "node_3", "to_port": "y_train"}
        ]
    }
    
    code = compile_pipeline(pipeline_data)
    
    assert "node_1_dataframe = load_csv(" in code
    assert "node_2_X_train," in code
    assert "node_3_model = train(" in code
    assert "import pandas as pd" in code
    assert "def main():" in code
    assert "return locals()" in code
    
    # Must compile correctly!
    try:
        parsed = ast.parse(code)
        assert parsed is not None
    except SyntaxError as e:
        pytest.fail(f"Compiled code is syntactically invalid: {e}\\n\\nCode:\\n{code}")


def test_compiler_unwired_inputs():
    """
    Test how the compiler treats required unwired inputs gracefully.
    """
    pipeline_data = {
        "nodes": [
            {
                "id": "node_1",
                "type": "train_classification",
                "params": {"algorithm": "random_forest"}
            }
        ],
        "connections": []
    }
    
    code = compile_pipeline(pipeline_data)
    assert "None" in code
    
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Compiled code is syntactically invalid: {e}\\n\\nCode:\\n{code}")


def test_compiler_full_execution():
    """
    Test that the generated code actually runs correctly locally 
    producing standard ADAMOPS outputs without failure.
    """
    import pandas as pd
    from sklearn.datasets import make_classification
    import traceback
    
    # Setup standard dummy file
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(4)])
    df["target"] = y
    
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f.name, index=False)
        filepath = f.name
        
    pipeline_data = {
        "nodes": [
            {
                "id": "read_node",
                "type": "load_csv",
                "params": {"filepath": filepath}
            },
            {
                "id": "split_node",
                "type": "train_test_split",
                "params": {"target": "target", "test_size": 0.2}
            },
            {
                "id": "train_node",
                "type": "train_classification",
                "params": {"algorithm": "random_forest"}
            },
            {
                "id": "eval_node",
                "type": "evaluate",
                "params": {}
            }
        ],
        "connections": [
            {"from_node": "read_node", "from_port": "dataframe", "to_node": "split_node", "to_port": "dataframe"},
            {"from_node": "split_node", "from_port": "X_train", "to_node": "train_node", "to_port": "X_train"},
            {"from_node": "split_node", "from_port": "y_train", "to_node": "train_node", "to_port": "y_train"},
            
            {"from_node": "split_node", "from_port": "X_test", "to_node": "eval_node", "to_port": "X_test"},
            {"from_node": "split_node", "from_port": "y_test", "to_node": "eval_node", "to_port": "y_test"},
            {"from_node": "train_node", "from_port": "model", "to_node": "eval_node", "to_port": "model"}
        ]
    }
    
    code = compile_pipeline(pipeline_data)
    
    # Setup isolated environment
    namespace = {}
    
    try:
        # Run the string output
        exec(code, namespace)
        assert "main" in namespace
        
        # Execute the auto-generated method
        results = namespace["main"]()
        
        # Validate that the resulting locals dictionary tracks expectations
        assert "eval_node_metrics" in results
        metrics = results["eval_node_metrics"]
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert "train_node_model" in results
        assert hasattr(results["train_node_model"], "predict")
        
    except Exception as e:
        extracted = traceback.format_exc()
        pytest.fail(f"Execution Error running compiled script: {e}\\n\\nTraceback:\\n{extracted}")
