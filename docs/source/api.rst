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
