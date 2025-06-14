API Reference
=============

This page contains the API reference for SynthX. The library is organized into several modules for different aspects of cardiovascular anatomy synthesis.

Models
------

VAE
~~~

.. py:class:: CoMA(config: Optional[Dict] = None)

   Bases: :py:class:`BaseGenerativeModel`

   CoMA: Convolutional Mesh Autoencoder implementation.

   This model implements the CoMA architecture as described in the paper "Generating 3D faces using convolutional mesh autoencoders" (Ranjan et al., 2018). CoMA is a variational autoencoder designed specifically for mesh data, using graph convolutional operations and specialized mesh pooling for cardiovascular anatomy synthesis.

   The model uses a hierarchical mesh representation with multiple resolution levels, where each level is connected through downsampling and upsampling transformations to capture both local and global anatomical features.

   **PARAMETERS:**

   * **config** (*Dict*, *optional*) - Configuration parameters for the model including:

     * **in_chan** (*int*) - Number of input channels (default: ``3``)
     * **out_chan** (*int*) - Number of output channels (default: ``3``) 
     * **feat_dims** (*List[int]*) - Feature dimensions for each layer (default: ``[32, 64, 128, 256]``)
     * **latent_dim** (*int*) - Dimension of the latent space (default: ``8``)
     * **block_config** (*Dict*) - Configuration for encoder/decoder blocks including:

       * **type** (*str*) - Block type (default: ``'CoMAConvBlock'``)
       * **conv_type** (*str*) - Type of graph convolution (default: ``'ChebConv'``)
       * **conv_kwargs** (*Dict*) - Arguments for convolution (default: ``{'K': 6}``)

     * **kl_weight** (*float*) - Weight for KL divergence loss (default: ``0.001``)
     * **mesh_topology** (*Dict*) - Mesh topology configuration including:

       * **edge_index_path** (*str*) - Path to edge indices file
       * **down_transforms_path** (*str*) - Path to downsampling transforms
       * **up_transforms_path** (*str*) - Path to upsampling transforms

   **SHAPES:**

   * **input**: mesh data (:math:`|V|, F_{in}`), where :math:`|V|` is the number of vertices and :math:`F_{in}` is the number of input features
   * **output**: reconstructed mesh data (:math:`|V|, F_{out}`)

   .. py:method:: forward(x: torch.Tensor) -> Dict[str, Any]

      Forward pass of the VAE.

      **PARAMETERS:**

      * **x** (*torch.Tensor*) - Input tensor of shape ``[batch_size, num_nodes, in_channels]``

      **RETURN TYPE:**
         *Dict[str, Any]*

      Returns dictionary with forward pass results including:
      
      * **x** - Original input
      * **recon_x** - Reconstructed output  
      * **q** - Latent distribution
      * **z** - Sampled latent vector

   .. py:method:: encode(x: torch.Tensor) -> Normal

      Encode input to latent distribution.

      **PARAMETERS:**

      * **x** (*torch.Tensor*) - Input tensor of shape ``[batch_size, num_nodes, in_channels]``

      **RETURN TYPE:**
         *torch.distributions.Normal*

   .. py:method:: decode(z: torch.Tensor) -> torch.Tensor

      Decode latent vector to output.

      **PARAMETERS:**

      * **z** (*torch.Tensor*) - Latent vector of shape ``[batch_size, latent_dim]``

      **RETURN TYPE:**
         *torch.Tensor*

      Returns reconstructed output of shape ``[batch_size, num_nodes, out_channels]``

   .. py:method:: generate(n_samples: int, **kwargs) -> torch.Tensor

      Generate synthetic mesh data by sampling from the latent space.

      **PARAMETERS:**

      * **n_samples** (*int*) - Number of samples to generate
      * **kwargs** - Additional generation parameters

      **RETURN TYPE:**
         *torch.Tensor*

      Returns generated mesh data of shape ``[n_samples, num_nodes, out_channels]``

   .. py:method:: loss_func(model_output: Dict[str, Any], target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]

      Calculate the VAE loss (reconstruction + KL divergence).

      **PARAMETERS:**

      * **model_output** (*Dict[str, Any]*) - Output from the model's forward pass
      * **target** (*torch.Tensor*) - Target data (typically the same as input)

      **RETURN TYPE:**
         *Tuple[torch.Tensor, Dict[str, float]]*

   .. py:method:: training_step(batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]

      Perform a single training step on a batch of data.

      **PARAMETERS:**

      * **batch** (*torch.Tensor*) - A batch of mesh data
      * **optimizer** (*torch.optim.Optimizer*) - The optimizer to use

      **RETURN TYPE:**
         *Dict[str, float]*

Utilities
---------

Trainer
~~~~~~~

.. py:class:: Trainer(config: Dict[str, Any])

   Simple trainer for generative models in the SynthX library.

   Handles training, validation, checkpointing, and logging for anatomy generation models. Provides a unified interface for training VAE models with configurable optimization and monitoring settings.

   **PARAMETERS:**

   * **config** (*Dict[str, Any]*) - Training configuration containing:

     * **epochs** (*int*) - Number of training epochs
     * **batch_size** (*int*) - Batch size for training
     * **device** (*str*) - Device for training (``"cpu"`` or ``"cuda"``)
     * **checkpoint_dir** (*str*) - Directory to save checkpoints
     * **save_every** (*int*) - Save checkpoint every N epochs
     * **save_best** (*bool*) - Whether to save best model
     * **log_every** (*int*) - Log metrics every N steps
     * **validate_every** (*int*) - Validate every N epochs
     * **optimizer_type** (*str*, *optional*) - Optimizer type (``"Adam"`` or ``"SGD"``, default: ``"Adam"``)
     * **optimizer_config** (*Dict*, *optional*) - Optimizer configuration including learning rate
     * **num_workers** (*int*, *optional*) - Number of data loader workers (default: ``0``)
     * **pin_memory** (*bool*, *optional*) - Whether to pin memory in data loaders (default: ``False``)

   .. py:method:: fit(model: nn.Module, train_dataset: Any, val_dataset: Optional[Any] = None) -> None

      Train the model on the provided datasets.

      **PARAMETERS:**

      * **model** (*nn.Module*) - The generative model to train
      * **train_dataset** (*Any*) - Training dataset
      * **val_dataset** (*Optional[Any]*) - Validation dataset

   .. py:method:: evaluate(dataset: Any, evaluator: Optional[Any] = None, metrics: Optional[list] = None) -> Dict[str, float]

      Evaluate the model on a dataset.

      **PARAMETERS:**

      * **dataset** (*Any*) - Dataset to evaluate on
      * **evaluator** (*Optional[Any]*) - Evaluator to use for metrics computation
      * **metrics** (*Optional[list]*) - List of specific metrics to compute

      **RETURN TYPE:**
         *Dict[str, float]*

   .. py:method:: load_checkpoint(checkpoint_path: str) -> None

      Load model checkpoint.

      **PARAMETERS:**

      * **checkpoint_path** (*str*) - Path to the checkpoint file

      **Raises:**
         * **FileNotFoundError** - If checkpoint file is not found

   .. py:method:: get_training_history() -> Dict[str, list]

      Get training history.

      **RETURN TYPE:**
         *Dict[str, list]*

      Returns training history containing losses and epochs.

Sampler
~~~~~~~

.. py:class:: Sampler(config: Dict)

   Simple sampler for generating virtual patients from trained generative models.

   This class provides a clean interface for sampling from generative models to create synthetic anatomical data with memory-efficient batch processing and configurable output formats.

   **PARAMETERS:**

   * **config** (*Dict*) - Configuration parameters including:

     * **batch_size** (*int*) - Default batch size for sampling
     * **n_samples** (*int*) - Number of samples to generate
     * **seed** (*int*) - Random seed for reproducible sampling
     * **save_path** (*str* or *Path*) - Path to save generated samples
     * **save_format** (*str*) - Format to save samples (e.g., ``"npy"``, ``"npz"``, ``"pt"``)

   .. py:method:: sample(model: BaseGenerativeModel, **kwargs) -> None

      Generate synthetic samples from the model.

      **PARAMETERS:**

      * **model** (*BaseGenerativeModel*) - Trained generative model
      * **kwargs** - Additional arguments passed to model.generate()

   .. py:method:: set_seed(seed: int) -> None
      :staticmethod:

      Set random seed for reproducible sampling.

      **PARAMETERS:**

      * **seed** (*int*) - Random seed value

Evaluators
----------

MeshEvaluator
~~~~~~~~~~~~~

.. py:class:: MeshEvaluator(config: Optional[Dict] = None)

   Bases: :py:class:`BaseEvaluator`

   Evaluator for mesh-based metrics including specificity and coverage.

   This evaluator compares virtual mesh populations against real reference mesh populations using configurable distance metrics. It provides a unified interface for computing various mesh evaluation metrics with flexible distance computation and data preprocessing options.

   **PARAMETERS:**

   * **config** (*Optional[Dict]*) - Configuration parameters including:

     * **dtype** (*str*, *optional*) - Data type for computations (``'float32'`` or ``'float64'``)
     * **virtual_population_path** (*str*, *optional*) - Path to virtual population data
     * **real_population_path** (*str*, *optional*) - Path to real population data
     * **[metric_name]** (*Dict*) - Configuration for specific metrics, each containing:

       * **distance** (*Dict*) - Distance metric configuration with:

         * **type** (*str*) - Distance metric type (default: ``'chamfer'``)
         * **bidirectional** (*bool*) - Whether to use bidirectional distance (default: ``True``)

       * Additional metric-specific parameters

   .. py:method:: evaluate(virtual_population: Any, real_population: Optional[Any] = None, metrics: Optional[List[str]] = None, **kwargs) -> Dict[str, float]

      Evaluate virtual mesh population using specified metrics.

      **PARAMETERS:**

      * **virtual_population** (*Any*) - Virtual population data (list of meshes)
      * **real_population** (*Optional[Any]*) - Real population data (list of meshes)
      * **metrics** (*Optional[List[str]]*) - List of metrics to compute. If ``None``, uses all available metrics
      * **kwargs** - Additional parameters

      **RETURN TYPE:**
         *Dict[str, float]*

      Returns dictionary of metric names and computed values.

   .. py:method:: compute_metric(metric_name: str, virtual_population: Any, real_population: Optional[Any] = None) -> float

      Compute a single metric value by name.

      **PARAMETERS:**

      * **metric_name** (*str*) - Name of the metric to compute
      * **virtual_population** (*Any*) - Virtual population data (list of meshes)
      * **real_population** (*Optional[Any]*) - Real population data (list of meshes)

      **RETURN TYPE:**
         *float*

      **Raises:**
         * **KeyError** - If metric is not registered
         * **ValueError** - If real_population is None

   .. py:method:: prepare_data(data: Any) -> Any

      Prepare mesh data for evaluation.

      **PARAMETERS:**

      * **data** (*Any*) - Input data to prepare

      **RETURN TYPE:**
         *Any*

      Returns prepared data with appropriate data type conversion.

Metrics
-------

Specificity
~~~~~~~~~~~

.. py:function:: compute_specificity(virtual_population: List[np.ndarray], real_population: List[np.ndarray], distance_metric: BaseDistance, center: bool = False, normalize: bool = False, **kwargs) -> float

   Compute specificity metric.

   Specificity is defined as the average distance of virtual patients to their nearest neighbors in the real population, as described in "Building 3-D statistical shape models by direct optimization" (Davies et al., 2009). This metric measures how realistic the synthetic samples are by evaluating how well they fit within the real population distribution.

   The specificity metric evaluates the quality of synthetic mesh generation by measuring the distance from each virtual sample to its closest real sample. Lower specificity values indicate that virtual samples are closer to real data, suggesting better generation quality.

   **PARAMETERS:**

   * **virtual_population** (*List[np.ndarray]*) - Virtual/synthetic population data. Each element is a point cloud of shape ``(Ni, 3)``
   * **real_population** (*List[np.ndarray]*) - Real/reference population data. Each element is a point cloud of shape ``(Mj, 3)``
   * **distance_metric** (*BaseDistance*) - Distance metric to use for computing nearest neighbors
   * **center** (*bool*, *optional*) - If ``True``, center each point cloud at origin (default: ``False``)
   * **normalize** (*bool*, *optional*) - If ``True``, normalize each point cloud to unit sphere (default: ``False``)
   * **kwargs** - Additional parameters (unused but kept for consistency)

   **RETURN TYPE:**
      *float*

   Returns the specificity value (average nearest neighbor distance). Lower values indicate better specificity.

   **SHAPES:**

   * **virtual_population**: List of :math:`N_{virtual}` point clouds, each of shape :math:`(N_i, 3)`
   * **real_population**: List of :math:`N_{real}` point clouds, each of shape :math:`(M_j, 3)`

Coverage
~~~~~~~~

.. py:function:: compute_coverage(virtual_population: List[np.ndarray], real_population: List[np.ndarray], distance_metric: BaseDistance, center: bool = False, normalize: bool = False, **kwargs) -> float

   Compute coverage metric.

   Coverage is defined as the proportion of real patients that have been identified as the nearest neighbor of at least one virtual patient, as described in "Learning representations and generative models for 3d point clouds" (Achlioptas et al., 2018). This metric measures how well the synthetic population covers the diversity present in the real population.

   The coverage metric evaluates the diversity of synthetic mesh generation by determining what percentage of real samples are represented by the virtual population. Higher coverage values indicate that the synthetic data better captures the full range of variation in the real data.

   **PARAMETERS:**

   * **virtual_population** (*List[np.ndarray]*) - Virtual/synthetic population data. Each element is a point cloud of shape ``(Ni, 3)``
   * **real_population** (*List[np.ndarray]*) - Real/reference population data. Each element is a point cloud of shape ``(Mj, 3)``
   * **distance_metric** (*BaseDistance*) - Distance metric to use for computing nearest neighbors
   * **center** (*bool*, *optional*) - If ``True``, center each point cloud at origin (default: ``False``)
   * **normalize** (*bool*, *optional*) - If ``True``, normalize each point cloud to unit sphere (default: ``False``)
   * **kwargs** - Additional parameters (unused but kept for consistency)

   **RETURN TYPE:**
      *float*

   Returns the coverage value as a percentage (0-100) of real samples covered by the virtual population. Higher values indicate better coverage.

   **SHAPES:**

   * **virtual_population**: List of :math:`N_{virtual}` point clouds, each of shape :math:`(N_i, 3)`
   * **real_population**: List of :math:`N_{real}` point clouds, each of shape :math:`(M_j, 3)`

Metrics
-------

Specificity
~~~~~~~~~~~

.. py:function:: compute_specificity(virtual_population: List[np.ndarray], real_population: List[np.ndarray], distance_metric: BaseDistance, center: bool = False, normalize: bool = False, **kwargs) -> float

   Compute specificity metric.

   Specificity is defined as the average distance of virtual patients to their nearest neighbors in the real population, as described in "Building 3-D statistical shape models by direct optimization" (Davies et al., 2009). This metric measures how realistic the synthetic samples are by evaluating how well they fit within the real population distribution.

   The specificity metric evaluates the quality of synthetic mesh generation by measuring the distance from each virtual sample to its closest real sample. Lower specificity values indicate that virtual samples are closer to real data, suggesting better generation quality.

   **PARAMETERS:**

   * **virtual_population** (*List[np.ndarray]*) - Virtual/synthetic population data. Each element is a point cloud of shape ``(Ni, 3)``
   * **real_population** (*List[np.ndarray]*) - Real/reference population data. Each element is a point cloud of shape ``(Mj, 3)``
   * **distance_metric** (*BaseDistance*) - Distance metric to use for computing nearest neighbors
   * **center** (*bool*, *optional*) - If ``True``, center each point cloud at origin (default: ``False``)
   * **normalize** (*bool*, *optional*) - If ``True``, normalize each point cloud to unit sphere (default: ``False``)
   * **kwargs** - Additional parameters (unused but kept for consistency)

   **RETURN TYPE:**
      *float*

   Returns the specificity value (average nearest neighbor distance). Lower values indicate better specificity.

   **SHAPES:**

   * **virtual_population**: List of :math:`N_{virtual}` point clouds, each of shape :math:`(N_i, 3)`
   * **real_population**: List of :math:`N_{real}` point clouds, each of shape :math:`(M_j, 3)`
