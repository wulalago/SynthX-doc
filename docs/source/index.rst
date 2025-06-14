Welcome to SynthX's documentation!
===================================

**SynthX** (/ˈsɪnθ-ɛks/) is a powerful Python library for synthesising virtual populations of cardiovascular anatomy. Designed for medical device development and cardiovascular research, SynthX provides the tools you need to generate realistic, statistically representative cardiovascular anatomical datasets.

What is SynthX?
---------------

SynthX enables researchers and medical device developers to create synthetic cardiovascular populations that maintain the statistical properties and variations found in real anatomical data, while providing the scale and control needed for robust computational studies.

Key Features
------------

* **Cardiovascular Focus**: Specialised models for heart, vessels, and cardiovascular anatomy
* **Population Synthesis**: Generate thousands of anatomically plausible cardiovascular variations
* **Statistical Validation**: Maintain realistic distributions and correlations from real data
* **Model Training**: Train generative models on your own cardiovascular datasets
* **Evaluation Metrics**: Built-in tools to validate synthetic population quality
* **Flexible Sampling**: Generate populations with specific demographic characteristics
* **Research Ready**: Export results for immediate use in medical device studies

Quick Start
-----------

Here's a simple example to get you started with SynthX:

.. code-block:: python

    import synthx
    import yaml

    # Load configuration
    with open('cardiovascular_config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Create and train a cardiovascular model
    model = synthx.build_model(config['model_type'], config['model_config'])
    train_dataset = synthx.build_dataset(config['dataset_type'], config['dataset_config'])
    
    trainer = synthx.Trainer(config=config)
    trainer.fit(model, train_dataset)

    # Generate synthetic cardiovascular population
    sampler = synthx.Sampler(config=config['sampler_config'])
    synthetic_population = sampler.sample(model, num_samples=1000)

    # Evaluate the quality of synthetic data
    evaluator = synthx.build_evaluator(config['evaluator_type'], config['evaluator_config'])
    results = evaluator.evaluate(synthetic_population, real_population_path)
    
    print(f"Population quality metrics: {results}")

Installation
------------

Get started with SynthX in just a few steps. Check out the :doc:`usage` section for detailed installation instructions, including how to :ref:`installation` the project.

Use Cases
---------

Medical Device Development
~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate diverse cardiovascular anatomies for comprehensive device testing without the need for large patient cohorts or privacy concerns.

Regulatory Validation
~~~~~~~~~~~~~~~~~~~~~
Create extensive test datasets that demonstrate device performance across population variations required for regulatory submissions.

Research Applications
~~~~~~~~~~~~~~~~~~~~~
Develop cardiovascular research studies with controlled anatomical parameters and unlimited sample sizes.

AI/ML Training
~~~~~~~~~~~~~~
Build robust machine learning models for cardiovascular applications with large, diverse training datasets.

Getting Started
---------------

New to SynthX? Follow these steps:

1. **Installation**: Set up SynthX in your Python environment
2. **API Reference**: Explore the complete technical documentation

.. note::
   This project is under active development. The current version focuses on cardiovascular anatomy synthesis. 
   Additional anatomical regions will be added in future releases.

.. warning::
   SynthX generates synthetic data for research and development purposes. Always validate results 
   with real data and follow appropriate regulatory guidelines for medical device applications.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   installation
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   modules/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   contributing
   license
   faq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

