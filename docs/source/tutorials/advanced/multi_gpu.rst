Training with Multiple GPUs
===========================

The basics demos call ``model.train(max_epochs=...)``, which is a thin wrapper
around ``lightning.Trainer``: every keyword argument is forwarded to the
``Trainer`` constructor. Multi-GPU training therefore only requires adding a
few extra kwargs to the same call — no other changes to the demo are needed,
except for two environment-level adjustments described below.

1. Adjust GPU visibility
------------------------

The basics demos start with::

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

This restricts the process to a single GPU and **must be changed for
multi-GPU training**. Either remove the line entirely (use every GPU on the
node), or list the GPUs you want to use::

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'   # use 4 GPUs

2. Switch the dataloader to the distributed sampler
---------------------------------------------------

Pass ``sampler_type='ddp'`` when configuring the data::

    from lightning import seed_everything
    seed_everything(42)

    model = MIDAS.configure_data_from_dir(
        configs=configs,
        dir_path='dataset/' + task + '/data',
        save_model_path='saved_models/' + task,
        sampler_type='ddp',
    )

3. Train with DDP
-----------------

Replace the demo's training call::

    model.train(max_epochs=1500)

with::

    model.train(
        max_epochs=1500,
        accelerator='gpu',
        devices='auto',     # use every visible GPU; or e.g. devices=2
        strategy='ddp',
    )

``devices`` accepts:

- ``'auto'`` — use every GPU made visible by ``CUDA_VISIBLE_DEVICES``
- an integer — number of GPUs (e.g. ``devices=2``)
- a list — explicit GPU IDs (e.g. ``devices=[0, 2]``)

.. note::
    DDP cannot be launched directly from a Jupyter notebook (Lightning
    requires a script entry point so each rank can re-import the same
    module). You have two options:

    1. **Recommended**: convert the notebook to a ``.py`` script::

           jupyter nbconvert --to script demo1.ipynb

       then run it with ``python demo1.py`` (Lightning will spawn one process
       per GPU automatically).

    2. **Notebook-friendly fallback**: use ``strategy='ddp_notebook'`` instead
       of ``'ddp'``. This launches DDP via ``torch.multiprocessing`` and works
       inside notebooks, but is slower to start and less commonly used in
       production.

.. note::
    Inference (``model.predict``, ``model.get_emb_umap``) is decorated with
    ``@rank_zero_only`` and runs on a single process, so the post-training
    cells in the demos do not need to be changed.
