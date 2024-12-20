Visualizing the Training Process (TensorBoard) 
===============================================

To visualize the training progress, you first need to integrate TensorBoard logging into the model in the training code snippet:

.. code-block:: python

    from lightning.pytorch import loggers as pl_loggers
    
    # Initialize TensorBoard logging
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='./logs/', 
        version='task_version'  # Replace with a descriptive version name
    )

    # Configure the trainer for single-device training
    trainer = L.Trainer(
        logger=tb_logger,              # Attach the logger
    )

Next, to monitor the training process, run the following command in your terminal:

.. code-block:: python

    tensorboard --logdir './logs/lightning_logs'

Finally, open the URL displayed in your terminal (e.g., http://localhost:6006) in a web browser to visualize the training loss and results.

.. figure:: ../../_static/img/tensorboard.png
   :alt: midas_structure.png
   :align: center