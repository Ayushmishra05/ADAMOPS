.. _getting_started:

Getting Started
===============

Installation
------------

You can install AdamOps and all of its extensive features directly from PyPI:

.. code-block:: bash

    pip install adamops

Using the Model Playground
--------------------------

AdamOps features a built-in Streamlit dashboard for instantaneously visualizing your trained models, evaluating metrics, and generating predictions.

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from adamops.deployment.playground import launch_playground

    # 1. Load data
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train a standard model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 3. Launch the playground to interact with your model!
    launch_playground(model=model, X_test=X_test, y_test=y_test)

The dashboard will open in your browser automatically, providing live sliding predictions, confusion matrices, and interactive data exploration over your test dataset.

Using AdamOps Studio
--------------------

If you prefer building ML workflows visually instead of writing code, you can use the **AdamOps Studio** pipeline builder. 

Launch the studio from your terminal:

.. code-block:: bash

    adamops studio

Or launch it directly from Python:

.. code-block:: python

    from adamops.studio import launch
    launch()

This commands opens a drag-and-drop web interface where you can assemble data loading, preprocessing, model training, and evaluation nodes into an executable pipeline graph.
